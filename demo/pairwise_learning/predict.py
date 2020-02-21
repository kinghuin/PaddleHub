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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    module._initialize()

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

    # Setup feed list for data feeder
    # Must feed all the tensor of module need

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.PairwiseTask(
        module=module,
        data_reader=reader,
        max_seq_len=args.max_seq_len,
        margin=0.1,
        nets_num=3,
        config=config,
    )

    # Data to be prdicted
    data = [["北京奥运博物馆的场景效果负责人是谁？", "主要承担奥运文物征集、保管、研究和爱国主义教育基地建设相关工作。"],
            ["北京奥运博物馆的场景效果负责人是谁", "于海勃，美国加利福尼亚大学教授 场景效果负责人 总设计师"],
            ["北京奥运博物馆的场景效果负责人是谁？", "洪麦恩，清华大学美术学院教授 内容及主展线负责人 总设计师"]]

    print(cls_task.predict(data=data, return_result=True))
