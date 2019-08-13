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
from paddlehub.finetune.evaluate import recall_nk

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--dataset", type=str, default="udc", help="Directory to model checkpoint")
parser.add_argument("--data_dir", type=str, default=None, help="Path to training data.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub ERNIE pretrained model
    #module = hub.Module(name="ernie")
    dataset = None
    # Download dataset and use ClassifyReader to read dataset
    if args.dataset.lower() == "udc":
        dataset = hub.dataset.UDC()
        module = hub.Module(name="bert_uncased_L-12_H-768_A-12")
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    inputs, outputs, program = module.context(trainable=True,
                                              max_seq_len=args.max_seq_len)

    reader = hub.reader.DialogReader(dataset=dataset,
                                     vocab_path=module.get_vocab_path(),
                                     max_seq_len=args.max_seq_len)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(use_data_parallel=False,
                           use_pyreader=args.use_pyreader,
                           use_cuda=args.use_gpu,
                           batch_size=args.batch_size,
                           enable_memory_optim=False,
                           checkpoint_dir=args.checkpoint_dir,
                           strategy=hub.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(data_reader=reader,
                                      feature=pooled_output,
                                      feed_list=feed_list,
                                      num_classes=dataset.num_labels,
                                      config=config)

    # Data to be prdicted
    data = [[d.text_a, d.text_b] for d in dataset.get_test_examples()]
    labels = np.array(([int(d.label) for d in dataset.get_test_examples()]))
    run_states = cls_task.predict(data=data)

    first = 1
    for run_state in run_states:
        for batch_result in run_state.run_results:
            if first:
                results = batch_result
                first = 0
            else:
                results = np.concatenate((results, batch_result), axis=0)

    infer_labels = np.argmax(results, axis=1)
    fout = open("predict.txt", "w")
    #     for i,j in enumerate(infer_labels):
    #         if i<3:
    #             print("text_a:%s\ttext_b%s\tpredict:%s\t"%(data[i][0],data[i][1],j))

    data = []
    for i in range(len(infer_labels)):
        data.append([results[i][1], labels[i]])
        fout.write("%s\n" % [results[i][1], labels[i]])

    print("Recall 1@10:%.5f" % recall_nk(data, 10, 1, 10))
    print("Recall 2@10:%.5f" % recall_nk(data, 10, 2, 10))
    print("Recall 5@10:%.5f" % recall_nk(data, 10, 5, 10))
