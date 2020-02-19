# coding:utf-8
#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import OrderedDict
import numpy as np
import paddle.fluid as fluid

import paddlehub as hub
from paddlehub.common.paddle_helper import clone_program
from paddlehub.finetune.evaluate import calculate_f1_np, matthews_corrcoef
from paddlehub.common import logger
from .base_task import BaseTask, RunConfig


class PairwiseTask(BaseTask):
    def __init__(
            self,
            module,
            data_reader,
            max_seq_len,
            margin,
            nets_num,
            config=None,
            metrics_choices="default",
    ):
        super(PairwiseTask, self).__init__(
            feed_list=None,
            data_reader=data_reader,
            main_program=None,
            startup_program=None,
            config=config,
            metrics_choices=metrics_choices)
        self.module = module
        self.margin = margin
        self.nets_num = nets_num
        self.max_seq_len = max_seq_len

    def _build_env(self):
        if self.env.is_inititalized:
            return
        self._build_env_start_event()
        self.env.is_inititalized = True
        self.env.main_program = clone_program(
            self._base_main_program, for_test=False)

        self.env.startup_program = fluid.Program()
        with fluid.program_guard(self.env.main_program,
                                 self._base_startup_program):
            with fluid.unique_name.guard(self.env.UNG):
                self.env.outputs = self._build_net()
                if self.is_train_phase:
                    self.env.loss = self._add_loss()
                    self.env.metrics = self._add_metrics()
                if self.is_test_phase:
                    self.env.labels = self._add_label()
                    self.env.metrics = self._add_metrics()

        if self.is_predict_phase or self.is_test_phase:
            self.env.main_program = clone_program(
                self.env.main_program, for_test=True)
            hub.common.paddle_helper.set_op_attr(
                self.env.main_program, is_test=True)

            if self.config.enable_memory_optim:
                for var_name in self.fetch_list:
                    var = self.env.main_program.global_block().vars[var_name]
                    var.persistable = True

        if self.is_train_phase:
            with fluid.program_guard(self.env.main_program,
                                     self._base_startup_program):
                with fluid.unique_name.guard(self.env.UNG):
                    self.scheduled_lr, self.max_train_steps = self.config.strategy.execute(
                        self.loss, self._base_data_reader, self.config,
                        self.device_count)

        if self.is_train_phase:
            loss_name = self.env.loss.name
        else:
            loss_name = None

        share_vars_from = self._base_compiled_program

        if not self.config.use_data_parallel:
            self.env.main_program_compiled = None
        else:
            self.env.main_program_compiled = fluid.CompiledProgram(
                self.env.main_program).with_data_parallel(
                    loss_name=loss_name,
                    share_vars_from=share_vars_from,
                    build_strategy=self.build_strategy)

        self.exe.run(self.env.startup_program)
        self._build_env_end_event()

    def _add_input(self):
        def add_input_ids(prefix):
            return fluid.layers.data(
                name=prefix + 'input_ids',
                shape=[-1, self.max_seq_len, 1],
                dtype='int64',
                lod_level=0)

        def add_position_ids(prefix):
            return fluid.layers.data(
                name=prefix + 'position_ids',
                shape=[-1, self.max_seq_len, 1],
                dtype='int64',
                lod_level=0)

        def add_segment_ids(prefix):
            return fluid.layers.data(
                name=prefix + 'segment_ids',
                shape=[-1, self.max_seq_len, 1],
                dtype='int64',
                lod_level=0)

        def add_input_mask(prefix):
            return fluid.layers.data(
                name=prefix + 'input_mask',
                shape=[-1, self.max_seq_len, 1],
                dtype='float32',
                lod_level=0)

        def add_task_ids(prefix):
            return fluid.layers.data(
                name=prefix + 'task_ids',
                shape=[-1, self.max_seq_len, 1],
                dtype='int64',
                lod_level=0)

        def add_pyreader(copies, pyreader_name):
            pyreader = fluid.layers.py_reader(
                capacity=50,
                shapes=[[-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1],
                        [-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1],
                        [-1, self.max_seq_len, 1]] * copies,
                dtypes=[
                    'int64',
                    'int64',
                    'int64',
                    'float32',
                    'int64',
                ] * copies,
                lod_levels=[0, 0, 0, 0, 0] * copies,
                name=pyreader_name,
                use_double_buffer=True)
            self.env.py_reader = pyreader
            # input_ids, segment_ids, position_ids, input_mask, task_ids,
            return fluid.layers.read_file(pyreader)

        if self.is_train_phase:
            if self.nets_num == 2:
                inputs = {
                    "query_pos_input_ids": None,
                    "query_pos_position_ids": None,
                    "query_pos_segment_ids": None,
                    "query_pos_input_mask": None,
                    "query_pos_task_ids": None,
                    "query_neg_input_ids": None,
                    "qury_neg_position_ids": None,
                    "qury_neg_segment_ids": None,
                    "qury_neg_input_mask": None,
                    "qury_neg_task_ids": None,
                }
                if self.config.use_pyreader:
                    all_intputs = add_pyreader(2, "train_reader")
                    for i, key in inputs.keys():
                        inputs[key] = all_intputs[i]
                else:
                    inputs = {
                        "query_pos_input_ids": add_input_ids("query_pos"),
                        "query_pos_position_ids": add_position_ids("query_pos"),
                        "query_pos_segment_ids": add_segment_ids("query_pos"),
                        "query_pos_input_mask": add_input_mask("query_pos"),
                        "query_pos_task_ids": add_task_ids("query_pos"),
                        "query_neg_input_ids": add_input_ids("qury_neg"),
                        "qury_neg_position_ids": add_position_ids("qury_neg"),
                        "qury_neg_segment_ids": add_segment_ids("qury_neg"),
                        "qury_neg_input_mask": add_input_mask("qury_neg"),
                        "qury_neg_task_ids": add_task_ids("qury_neg"),
                    }

            elif self.nets_num == 3:
                inputs = {
                    "query_input_ids": None,
                    "query_position_ids": None,
                    "query_segment_ids": None,
                    "query_input_mask": None,
                    "query_task_ids": None,
                    "pos_input_ids": None,
                    "pos_position_ids": None,
                    "pos_segment_ids": None,
                    "pos_input_mask": None,
                    "pos_task_ids": None,
                    "neg_input_ids": None,
                    "neg_position_ids": None,
                    "neg_segment_ids": None,
                    "neg_input_mask": None,
                    "neg_task_ids": None,
                }
                if self.config.use_pyreader:
                    all_intputs = add_pyreader(3, "train_reader")
                    for i, key in enumerate(inputs.keys()):
                        inputs[key] = all_intputs[i]
                else:
                    inputs = {
                        "query_input_ids": add_input_ids("query"),
                        "query_position_ids": add_position_ids("query"),
                        "query_segment_ids": add_segment_ids("query"),
                        "query_input_mask": add_input_mask("query"),
                        "query_task_ids": add_task_ids("query"),
                        "pos_input_ids": add_input_ids("pos"),
                        "pos_position_ids": add_position_ids("pos"),
                        "pos_segment_ids": add_segment_ids("pos"),
                        "pos_input_mask": add_input_mask("pos"),
                        "pos_task_ids": add_task_ids("pos"),
                        "neg_input_ids": add_input_ids("neg"),
                        "neg_position_ids": add_position_ids("neg"),
                        "neg_segment_ids": add_segment_ids("neg"),
                        "neg_input_mask": add_input_mask("neg"),
                        "neg_task_ids": add_task_ids("neg"),
                    }

        else:
            if self.nets_num == 2:
                inputs = {
                    "query_pos_input_ids": None,
                    "query_pos_position_ids": None,
                    "query_pos_segment_ids": None,
                    "query_pos_input_mask": None,
                    "query_pos_task_ids": None
                }
                if self.config.use_pyreader:
                    all_intputs = add_pyreader(1, "train_reader")
                    for i, key in inputs.keys():
                        inputs[key] = all_intputs[i]
                else:
                    inputs = {
                        "query_pos_input_ids": add_input_ids("query_pos"),
                        "query_pos_position_ids": add_position_ids("query_pos"),
                        "query_pos_segment_ids": add_segment_ids("query_pos"),
                        "query_pos_input_mask": add_input_mask("query_pos"),
                        "query_pos_task_ids": add_task_ids("query_pos"),
                    }
            elif self.nets_num == 3:
                inputs = {
                    "query_input_ids": None,
                    "query_position_ids": None,
                    "query_segment_ids": None,
                    "query_input_mask": None,
                    "query_task_ids": None,
                    "pos_input_ids": None,
                    "pos_position_ids": None,
                    "pos_segment_ids": None,
                    "pos_input_mask": None,
                    "pos_task_ids": None,
                }
                if self.config.use_pyreader:
                    all_intputs = add_pyreader(2, "train_reader")
                    for i, key in inputs.keys():
                        inputs[key] = all_intputs[i]
                else:
                    inputs = {
                        "query_input_ids": add_input_ids("query"),
                        "query_position_ids": add_position_ids("query"),
                        "query_segment_ids": add_segment_ids("query"),
                        "query_input_mask": add_input_mask("query"),
                        "query_task_ids": add_task_ids("query"),
                        "pos_input_ids": add_input_ids("pos"),
                        "pos_position_ids": add_position_ids("pos"),
                        "pos_segment_ids": add_segment_ids("pos"),
                        "pos_input_mask": add_input_mask("pos"),
                        "pos_task_ids": add_task_ids("pos"),
                    }

        self.env.inputs = inputs
        return inputs

    def _build_net(self):
        inputs = self._add_input()
        if self.nets_num == 2:
            query_pos_pooled_output, _ = self.module.net(
                inputs["query_pos_input_ids"], inputs["query_pos_position_ids"],
                inputs["query_pos_segment_ids"], inputs["query_pos_input_mask"],
                inputs["query_pos_task_ids"])
            self.query_pos_sim = query_pos_pooled_output
            if self.is_train_phase:
                query_neg_pooled_output, _ = self.module.net(
                    inputs["query_neg_input_ids"],
                    inputs["qury_neg_position_ids"],
                    inputs["qury_neg_segment_ids"],
                    inputs["qury_neg_input_mask"], inputs["qury_neg_task_ids"])
                self.query_neg_sim = query_neg_pooled_output

        elif self.nets_num == 3:
            query_pooled_output, _ = self.module.net(
                inputs["query_input_ids"], inputs["query_position_ids"],
                inputs["query_segment_ids"], inputs["query_input_mask"],
                inputs["query_task_ids"])
            pos_pooled_output, _ = self.module.net(
                inputs["pos_input_ids"], inputs["pos_position_ids"],
                inputs["pos_segment_ids"], inputs["pos_input_mask"],
                inputs["pos_task_ids"])
            self.query_pos_sim = fluid.layers.cos_sim(query_pooled_output,
                                                      pos_pooled_output)
            if self.is_train_phase:
                neg_pooled_output, _ = self.module.net(
                    inputs["neg_input_ids"], inputs["neg_position_ids"],
                    inputs["neg_segment_ids"], inputs["neg_input_mask"],
                    inputs["neg_task_ids"])
                self.query_neg_sim = fluid.layers.cos_sim(
                    query_pooled_output, neg_pooled_output)
        if self.is_train_phase:
            return [self.query_pos_sim, self.query_neg_sim]
        else:
            return [self.query_pos_sim]

    def _add_label(self):
        return [fluid.layers.data(name="label", dtype="int64", shape=[1])]

    def _add_loss(self):
        neg_sub_pos = fluid.layers.elementwise_sub(self.query_neg_sim,
                                                   self.query_pos_sim)
        margin = fluid.layers.fill_constant_batch_size_like(
            self.query_neg_sim, self.query_neg_sim.shape, "float32",
            self.margin)
        add_margin = fluid.layers.elementwise_add(neg_sub_pos, margin)
        zero = fluid.layers.fill_constant_batch_size_like(
            self.query_neg_sim, self.query_neg_sim.shape, "float32", 0.0)
        max_margin = fluid.layers.elementwise_max(add_margin, zero)
        loss = fluid.layers.reduce_mean(max_margin)
        return loss

    def _add_metrics(self):
        if self.is_train_phase:
            # todo
            return []
        elif self.is_test_phase:
            acc = fluid.layers.accuracy(
                input=self.outputs[0], label=self.labels[0])
            return [acc]

    @property
    def feed_list(self):
        feed_list = [var.name for var in self.env.inputs.values()]
        if self.is_test_phase:
            feed_list += [label.name for label in self.labels]
        return feed_list

    @property
    def fetch_list(self):
        if self.is_test_phase:
            return [self.labels[0].name, self.query_pos_sim.name
                    ] + [metric.name for metric in self.metrics]
        elif self.is_train_phase:
            return [self.query_pos_sim.name, self.query_neg_sim
                    ] + [metric.name
                         for metric in self.metrics] + [self.loss.name]
        else:
            return [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        loss_sum = acc_sum = run_examples = 0
        run_step = run_time_used = 0
        all_labels = np.array([])
        all_infers = np.array([])

        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            if self.is_train_phase:
                loss_sum += np.mean(
                    run_state.run_results[-1]) * run_state.run_examples
            if self.is_test_phase:
                acc_sum += np.mean(
                    run_state.run_results[3]) * run_state.run_examples
                np_labels = run_state.run_results[0]
                np_infers = run_state.run_results[1]
                np_infers = (np_infers + 1) / 2
                all_labels = np.hstack((all_labels, np_labels.reshape([-1])))
                all_infers = np.hstack((all_infers, np_infers.reshape([-1])))

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        if self.is_test_phase:
            for metric in self.metrics_choices:
                if metric == "acc":
                    avg_acc = acc_sum / run_examples
                    scores["acc"] = avg_acc
                elif metric == "f1":
                    f1 = calculate_f1_np(all_infers, all_labels)
                    scores["f1"] = f1
                elif metric == "matthews":
                    matthews = matthews_corrcoef(all_infers, all_labels)
                    scores["matthews"] = matthews
                else:
                    raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed

    def _postprocessing(self, run_states):
        # todo
        try:
            id2label = {
                val: key
                for key, val in self._base_data_reader.label_map.items()
            }
        except:
            raise Exception(
                "ImageClassificationDataset does not support postprocessing, please use BaseCVDataset instead"
            )
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results
            batch_infer = np.argmax(batch_result, axis=2)[0]
            results += [id2label[sample_infer] for sample_infer in batch_infer]
        return results
