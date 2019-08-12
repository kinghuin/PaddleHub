#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import os
import math
import multiprocessing

import paddle.fluid as fluid

from paddlehub.common.logger import logger
from paddlehub.finetune.optimization import adam_weight_decay_optimization
from paddlehub.finetune.regularizer import L2SPDecayRegularizer
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow


def get_pretrained_parameter(main_program, start_program):
    pretrained_parameters = []
    global_block = main_program.global_block()
    for op in global_block.ops[::-1]:
        for input_arg in op.input_arg_names:
            var = global_block.var(input_arg)
            if isinstance(
                    var, fluid.framework.Parameter
            ) and input_arg not in start_program.global_block().vars:
                pretrained_parameters.append(var)

    return pretrained_parameters


def get_parentOp_depth_max(parent_ops, op_depth_dict):
    max_depth = 1
    for parent_op in parent_ops:
        depth = op_depth_dict.get(parent_op, 1)
        if max_depth < depth:
            max_depth = depth
    return max_depth


def get_opDepth_min(ops, op_depth_dict):
    min_depth = max(op_depth_dict.values())
    for op in ops:
        depth = op_depth_dict[op]
        if min_depth > depth:
            min_depth = depth
    return min_depth


def get_depth_parameter(main_program):
    pretrained_parameters = []
    global_block = main_program.global_block()

    var_op_dict = {}
    for op in global_block.ops:

        for input_arg in op.input_arg_names:
            if input_arg not in var_op_dict.keys():
                var_op_dict[input_arg] = {"output_ops": [], "input_ops": []}
            var_op_dict[input_arg]["output_ops"].append(op)

        for output_arg in op.output_arg_names:
            if output_arg not in var_op_dict.keys():
                var_op_dict[output_arg] = {"output_ops": [], "input_ops": []}
            var_op_dict[output_arg]["input_ops"].append(op)

    op_depth_dict = {}
    #     print(len(global_block.ops))
    for op in global_block.ops:
        parent_ops = []
        for input_arg in op.input_arg_names:
            for parent_op in var_op_dict[input_arg]["input_ops"]:
                if parent_op not in parent_ops:
                    parent_ops.append(parent_op)
        if not parent_ops:
            op_depth_dict[op] = 1
        else:
            op_depth_dict[op] = get_parentOp_depth_max(parent_ops,
                                                       op_depth_dict) + 1

    depth_params_dict = {}
    updated_depth_params_dict = {}
    for param in global_block.iter_parameters():
        adherent_ops = var_op_dict[param.name]["output_ops"]
        depth = get_opDepth_min(adherent_ops, op_depth_dict)
        if depth not in depth_params_dict.keys():
            depth_params_dict[depth] = []
            updated_depth_params_dict[depth] = []
        depth_params_dict[depth].append(param)
        updated_depth_params_dict[depth].append(param)

    depth_list = sorted(depth_params_dict.keys())
    len_depth_list = len(depth_list)
    for index, depth in enumerate(depth_list):
        for param in depth_params_dict[depth]:
            prefix = param.name.split(".")[0]
            if index < len_depth_list - 1:
                next_depth = depth_list[index + 1]
                for param_next_depth in depth_params_dict[next_depth]:
                    prefix_next_depth = param_next_depth.name.split(".")[0]
                    if prefix == prefix_next_depth:
                        updated_depth_params_dict[depth].append(
                            param_next_depth)
                        updated_depth_params_dict[next_depth].remove(
                            param_next_depth)

                        if not updated_depth_params_dict[next_depth]:
                            updated_depth_params_dict.pop(next_depth)

    return updated_depth_params_dict


def set_discriminative_learning_rate(main_program,
                                     max_learning_rate,
                                     num_abstract_blocks=3,
                                     lr_factor=2.6):
    depth_params_dict = get_depth_parameter(main_program)

    sorted_depth = sorted(depth_params_dict.keys(), reverse=True)
    _num_layers = math.ceil(len(sorted_depth) / num_abstract_blocks)

    power = 1
    cnt = 0
    for depth in sorted_depth:
        for index, param in enumerate(depth_params_dict[depth]):
            if depth_params_dict[depth][index].optimize_attr[
                    "learning_rate"] == 1.0:
                depth_params_dict[depth][index].optimize_attr[
                    "learning_rate"] = pow(1.0 / lr_factor, power)
            print(depth, param.optimize_attr)
        cnt += 1
        if cnt >= _num_layers:
            power += 1
            cnt = 0


def set_gradual_unfreeze(main_program, unfreeze_depths):
    depth_params_dict = get_depth_parameter(main_program)

    for depth in unfreeze_depths:
        for index, param in enumerate(depth_params_dict[depth]):
            depth_params_dict[depth][index].stop_gradient = False

    freeze_depths = list(
        set(depth_params_dict.keys()).difference(set(unfreeze_depths)))
    for depth in freeze_depths:
        for index, param in enumerate(depth_params_dict[depth]):
            depth_params_dict[depth][index].stop_gradient = True


def slanted_triangle_learning_rate_decay(
        cut_step, max_train_step, max_learning_rate, ratio, main_program):

    with main_program._lr_schedule_guard():
        global_step = lr_scheduler._decay_step_counter()

        lr = fluid.layers.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        with control_flow.Switch() as switch:
            with switch.case(global_step <= cut_step):
                pct = global_step / cut_step
                decayed_lr = max_learning_rate * (1 + pct * (ratio - 1)) / ratio
                fluid.layers.assign(decayed_lr, lr)
            with switch.default():
                pct = 1 - (global_step - cut_step) / (max_train_step - cut_step)
                decayed_lr = max_learning_rate * (1 + pct * (ratio - 1)) / ratio
                fluid.layers.assign(decayed_lr, lr)

        return lr


def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name.lower() == "sgd":
        optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adagrad":
        optimizer = fluid.optimizer.Adagrad(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adamax":
        optimizer = fluid.optimizer.Adamax(learning_rate=learning_rate)
    elif optimizer_name.lower() == "decayedadagrad":
        optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=learning_rate)
    elif optimizer_name.lower() == "ftrl":
        optimizer = fluid.optimizer.Ftrl(learning_rate=learning_rate)
    elif optimizer_name.lower() == "larsmomentum":
        optimizer = fluid.optimizer.LarsMomentum(learning_rate=learning_rate)
    elif optimizer_name.lower() == "momentum":
        optimizer = fluid.optimizer.Momentum(learning_rate=learning_rate)
    elif optimizer_name.lower() == "decayedadagrad":
        optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = fluid.optimizer.RMSPropOptimizer(
            learning_rate=learning_rate)
    else:
        optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)

    return optimizer


class DefaultStrategy(object):
    def __init__(self, learning_rate=1e-4, optimizer_name="adam"):
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        if self._optimizer_name.lower() == "sgd":
            self.optimizer = fluid.optimizer.SGD(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "adagrad":
            self.optimizer = fluid.optimizer.Adagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "adamax":
            self.optimizer = fluid.optimizer.Adamax(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "ftrl":
            self.optimizer = fluid.optimizer.Ftrl(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "larsmomentum":
            self.optimizer = fluid.optimizer.LarsMomentum(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "momentum":
            self.optimizer = fluid.optimizer.Momentum(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate)
        elif self._optimizer_name.lower() == "rmsprop":
            self.optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=self.learning_rate)
        else:
            self.optimizer = fluid.optimizer.Adam(
                learning_rate=self.learning_rate)

    def execute(self, loss, data_reader, config):
        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultStrategy's optimizer is None")

    # TODO complete __str__()
    def __str__(self):
        return "DefaultStrategy"

    def step(self):
        pass


class AdamWeightDecayStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 lr_scheduler="linear_decay",
                 warmup_proportion=0.1,
                 weight_decay=0.01,
                 optimizer_name="adam"):
        super(AdamWeightDecayStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        # check strategy correctness
        if lr_scheduler not in ["linear_decay", "noam_decay"]:
            raise ValueError("lr_scheduler {} is not setup "
                             "correctly".format(lr_scheduler))
        self._lr_scheduler = lr_scheduler
        self._warmup_proportion = warmup_proportion
        self._weight_decay = weight_decay

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def warmup_proportion(self):
        return self._warmup_proportion

    @property
    def weight_decay(self):
        return self._weight_decay

    def execute(self, loss, data_reader, config):
        main_program = loss.block.program
        # calculate wamrup step
        dev_count = self._get_dev_count(config)
        num_train_examples = data_reader.dataset.num_examples["train"]
        max_train_steps = config.num_epoch * num_train_examples // config.batch_size // dev_count
        warmup_steps = int(max_train_steps * self.warmup_proportion)

        scheduled_lr = adam_weight_decay_optimization(
            loss, warmup_steps, max_train_steps, self.learning_rate,
            main_program, self.weight_decay, self.lr_scheduler)

        return scheduled_lr

    def _get_dev_count(self, config):
        if config.use_cuda:
            dev_count = fluid.core.get_cuda_device_count()
        else:
            dev_count = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        return dev_count

    # TODO complete __str__()
    def __str__(self):
        return "AdamWeightDecayStrategy"


class DefaultFinetuneStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3):
        super(DefaultFinetuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        self.regularization_coeff = regularization_coeff

    def execute(self, loss, data_reader, config):
        # get pretrained parameters
        program = loss.block.program
        global_block = program.global_block()
        pretrained_params = get_pretrained_parameter(
            program, fluid.default_startup_program())

        # set parameter attrs
        for index, param in enumerate(pretrained_params):
            param.regularizer = fluid.regularizer.L2Decay(
                regularization_coeff=self.regularization_coeff)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultFinetuneStrategy's optimizer is None")


class L2SPFinetuneStrategy(DefaultStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3):
        super(L2SPFinetuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        self.regularization_coeff = regularization_coeff

    def execute(self, loss, data_reader, config):
        # get pretrained parameters
        program = loss.block.program
        global_block = program.global_block()
        pretrained_params = get_pretrained_parameter(
            program, fluid.default_startup_program())

        # set parameter attrs
        for index, param in enumerate(pretrained_params):
            param.regularizer = L2SPDecayRegularizer(
                regularization_coeff=self.regularization_coeff)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultFinetuneStrategy's optimizer is None")


class SlantedTriangleLRFineTuneStrategy(DefaultStrategy):
    def __init__(self,
                 ratio=32,
                 cut_fraction=0.1,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 use_gradual_unfreeze=True):
        super(SlantedTriangleLRFineTuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self._max_learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        self._ratio = ratio
        self._cut_fraction = cut_fraction
        self.epoch = 0
        self.use_gradual_unfreeze = use_gradual_unfreeze

    def step(self):
        self.epoch += 1

        depth_params_dict = get_depth_parameter(self.main_program)
        sorted_depth = sorted(depth_params_dict.keys(), reverse=True)
        max_depth = len(sorted_depth)

        if max_depth > 0:
            set_gradual_unfreeze(
                self.main_program, unfreeze_depths=sorted_depth[:self.epoch])
        else:
            logger.warning(
                "The max op-depth in the network is %s. That results in that can't use the gradual unfreeze finetune strategy."
                % (max_depth))

    @property
    def ratio(self):
        return self._ratio

    @property
    def cut_fraction(self):
        return self._cut_fraction

    @property
    def max_learning_rate(self):
        return self._max_learning_rate

    def execute(self, loss, data_reader, config):
        self.main_program = loss.block.program
        num_train_examples = data_reader.dataset.num_examples["train"]
        max_train_step = config.num_epoch * num_train_examples // config.batch_size
        cut_step = int(max_train_step * self.cut_fraction)
        scheduled_lr = slanted_triangle_learning_rate_decay(
            cut_step, max_train_step, self.max_learning_rate, self.ratio,
            self.main_program)
        self.optimizer = get_optimizer(self._optimizer_name, scheduled_lr)
        self.optimizer.minimize(loss)

        return scheduled_lr


class DiscriminativeLRFineTuneStrategy(DefaultStrategy):
    def __init__(self, learning_rate=1e-4, optimizer_name="adam",
                 lr_factor=2.6):
        super(DiscriminativeLRFineTuneStrategy, self).__init__(
            learning_rate=learning_rate, optimizer_name=optimizer_name)
        self._max_learning_rate = learning_rate
        self._lr_factor = lr_factor

    @property
    def lr_factor(self):
        return self._lr_factor

    @property
    def max_learning_rate(self):
        return self._max_learning_rate

    def execute(self, loss, data_reader=None, config=None):
        main_program = loss.block.program

        set_discriminative_learning_rate(main_program, self.max_learning_rate,
                                         4, self.lr_factor)

        if self.optimizer is not None:
            self.optimizer.minimize(loss)
