#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Finetuning on sequence labeling task."""

import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use SequenceLabelReader to read dataset
    dataset = hub.dataset.MSRA_NER()
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    # Compared to classification task, we need add seq_len tensor to feedlist
    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name
    ]

    # Select a finetune strategy
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler="linear_decay",
    )

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a sequence labeling finetune task by PaddleHub's API
    # if add crf, the network use crf as decoder
    seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        add_crf=True)

    cls_task = seq_label_task

    from paddlehub.common.logger import logger

    def func(self, run_states):
        scores, avg_loss, run_speed = self._calculate_metrics(run_states)
        self.tb_writer.add_scalar(
            tag="Loss_{}".format(self.phase),
            scalar_value=avg_loss,
            global_step=self._envs['train'].current_step)
        log_scores = ""
        for metric in scores:
            self.tb_writer.add_scalar(
                tag="{}_{}".format(metric, self.phase),
                scalar_value=scores[metric],
                global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, scores[metric])
        logger.info("NEW DEFAULT step %d / %d: loss=%.5f %s[step/sec: %.2f]" %
                    (self.current_step, self.max_train_steps, avg_loss,
                     log_scores, run_speed))

    def func1(self, run_states):
        logger.info("modify modify modify self.best_score=%s len(run_states)=%s"
                    % (self.best_score, len(run_states)))

    def func2(self, run_states):
        logger.info("add2 add2 add2 self.best_score=%s len(run_states)=%s" %
                    (self.best_score, len(run_states)))

    cls_task.delete_hook("log_interval", "default")
    print(cls_task.hooks)
    cls_task.add_hook("log_interval", "add", func)
    cls_task.add_hook("log_interval", "modified", func)
    print(cls_task.hooks)
    cls_task.modify_hook("log_interval", "modified", func1)
    cls_task.add_hook("log_interval", func2)
    try:
        cls_task.add_hook("log_interval", 000, func)
    except Exception as e:
        print(e)
    try:
        cls_task.add_hook("log_interval", 111)
    except Exception as e:
        print(e)
    try:
        cls_task.add_hook("log_interval")
    except Exception as e:
        print(e)
    try:
        cls_task.add_hook("xxx_interval", func)
    except Exception as e:
        print(e)
    try:
        cls_task.delete_hook("xxx_interval", "xxx")
    except Exception as e:
        print(e)
    try:
        cls_task.delete_hook("log_interval", "xxx")
    except Exception as e:
        print(e)
    try:
        cls_task.modify_hook("log_interval", "xxx", func)
    except Exception as e:
        print(e)
    try:
        cls_task.modify_hook("log_interval", "modified", 111)
    except Exception as e:
        print(e)

    # Finetune and evaluate model by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    seq_label_task.finetune_and_eval()