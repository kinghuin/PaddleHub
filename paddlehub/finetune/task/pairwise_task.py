#coding:utf-8
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

from paddlehub.finetune.evaluate import calculate_f1_np, matthews_corrcoef
from .base_task import BaseTask


class PairwiseTask(BaseTask):
    def __init__(self,
                 feature,
                 num_classes,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default",
                 margin=0.1):
        if metrics_choices == "default":
            metrics_choices = ["acc"]

        main_program = feature.block.program
        super(PairwiseTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self.feature = feature
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.margin = margin

    def _build_net(self):
        if len(self.feature) == 2:
            query_pos_pooled_output = self.feature['query_pos_pooled_output']
            query_neg_pooled_output = self.feature['query_neg_pooled_output']
            query_pos_cls_feats = fluid.layers.dropout(
                x=query_pos_pooled_output,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            query_neg_cls_feats = fluid.layers.dropout(
                x=query_neg_pooled_output,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            self.query_pos_sim = query_pos_cls_feats
            self.query_neg_sim = query_neg_cls_feats

        elif len(self.feature) == 3:
            query_pooled_output = self.feature['query_pooled_output']
            pos_pooled_output = self.feature['pos_pooled_output']
            neg_pooled_output = self.feature['neg_pooled_output']
            query_cls_feats = fluid.layers.dropout(
                x=query_pooled_output,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            pos_cls_feats = fluid.layers.dropout(
                x=pos_pooled_output,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            neg_cls_feats = fluid.layers.dropout(
                x=neg_pooled_output,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")

            self.query_pos_sim = fluid.layers.cos_sim(query_cls_feats,
                                                      pos_cls_feats)
            self.query_neg_sim = fluid.layers.cos_sim(query_cls_feats,
                                                      neg_cls_feats)

        return [self.query_pos_sim, self.query_neg_sim]

    def _add_label(self):
        if self.is_test_phase:
            return [fluid.layers.data(name="label", dtype="int64", shape=[1])]
        else:
            pass

    def _add_loss(self):
        if self.is_train_phase:
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
        else:
            ce_loss = fluid.layers.cross_entropy(
                input=self.outputs[0], label=self.labels[0])
            return fluid.layers.mean(x=ce_loss)

    def _add_metrics(self):
        if self.is_test_phase:
            acc = fluid.layers.accuracy(
                input=self.outputs[0], label=self.labels[0])
            return [acc]
        else:
            pass
            # TODO : WHAT?

    @property
    def fetch_list(self):
        if self.is_test_phase:
            return [self.labels[0].name, self.ret_infers.name
                    ] + [metric.name
                         for metric in self.metrics] + [self.loss.name]
        elif self.is_train_phase:
            return [self.loss.name]
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
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples
            if self.is_test_phase:
                acc_sum += np.mean(
                    run_state.run_results[2]) * run_state.run_examples
                np_labels = run_state.run_results[0]
                np_infers = run_state.run_results[1]
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
