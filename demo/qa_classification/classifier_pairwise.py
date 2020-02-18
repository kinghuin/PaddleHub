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

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, module_program = module.context(
        trainable=True, max_seq_len=args.max_seq_len, nets_num=3)

    # Download dataset and use accuracy as metrics
    # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
    # metric should be acc, f1 or matthews
    dataset = hub.dataset.NLPCC_DBQA_pairwise()

    # For ernie_tiny, it use sub-word to tokenize chinese sentence
    # If not ernie tiny, sp_model_path and word_dict_path should be set None
    reader = hub.reader.PairwiseReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path(),
        nets_num=3,
        sampling_rate=0.5)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = [
        outputs["query_pooled_output"], outputs["pos_pooled_output"],
        outputs["neg_pooled_output"]
    ]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["query_input_ids"].name,
        inputs["query_position_ids"].name,
        inputs["query_segment_ids"].name,
        inputs["query_input_mask"].name,
        inputs["pos_input_ids"].name,
        inputs["pos_position_ids"].name,
        inputs["pos_segment_ids"].name,
        inputs["pos_input_mask"].name,
        inputs["neg_input_ids"].name,
        inputs["neg_position_ids"].name,
        inputs["neg_segment_ids"].name,
        inputs["neg_input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.PairwiseTask(
        data_reader=reader,
        main_program=module_program,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        margin=0.1,
    )

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    cls_task.finetune_and_eval()
