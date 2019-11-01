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
"""Finetuning on classification task """

import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

hub.common.logger.logger.setLevel("INFO")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=384, help="Number of words of the longest seqence.")
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0, help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--n_best_size",  type=int, default=20,help="The total number of n-best predictions to generate in the ""nbest_predictions.json output file.")
parser.add_argument("--max_answer_length",  type=int, default=30,help="The maximum length of an answer that can be generated. This is needed ""because the start and end predictions are not conditioned on one another.")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=True, help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
parser.add_argument("--version_2_with_negative", type=ast.literal_eval, default=False, help="If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, it should be set true.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub bert_uncased_L-12_H-768_A-12 pretrained model
    module = hub.Module(name="bert_uncased_L-12_H-768_A-12")

    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use ReadingComprehensionReader to read dataset
    dataset = hub.dataset.SQUAD(
        version_2_with_negative=args.version_2_with_negative)

    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_length=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

    seq_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        lr_scheduler="linear_decay")

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        log_interval=10,
        use_pyreader=args.use_pyreader,
        use_data_parallel=args.use_data_parallel,
        save_ckpt_interval=1000,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        enable_memory_optim=True,
        strategy=strategy)

    # Define a reading comprehension finetune task by PaddleHub's API
    sub_task = "squad" if not args.version_2_with_negative else "squad2.0"
    reading_comprehension_task = hub.ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config,
        sub_task=sub_task,
    )

    # Finetune by PaddleHub's API
    reading_comprehension_task.finetune()
