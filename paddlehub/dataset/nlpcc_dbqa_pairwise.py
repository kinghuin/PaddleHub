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

import codecs
import os
import csv

from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset import InputExample
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/nlpcc-dbqa.tar.gz"


class PairwiseExample(object):
    """
    Input data structure of BERT/ERNIE, can satisfy single sequence task like
    text classification, sequence lableing; Sequence pair task like dialog
    task.
    """

    def __init__(self, guid, qid, query, pos, neg):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.qid = qid
        self.query = query
        self.pos = pos
        self.neg = neg

    def __str__(self):
        return "query={}\tpos={},neg={}".format(self.query, self.pos, self.neg)


class NLPCC_DBQA_pairwise(BaseNLPDataset):
    """
    Please refer to
    http://tcci.ccf.org.cn/conference/2017/dldoc/taskgline05.pdf
    for more information
    """

    def __init__(self):
        dataset_dir = os.path.join(DATA_HOME, "nlpcc-dbqa-pairwise")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(NLPCC_DBQA_pairwise, self).__init__(
            base_path=base_path,
            train_file="train.tsv",
            dev_file="dev.tsv",
            test_file="test.tsv",
            label_file=None,
            label_list=["0", "1"],
        )

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                if phase == "train":
                    example = PairwiseExample(
                        guid=seq_id,
                        qid=line[0],
                        query=line[1],
                        pos=line[2],
                        neg=line[3])
                else:
                    example = InputExample(
                        guid=seq_id,
                        label=line[3],
                        text_a=line[1],
                        text_b=line[2])
                seq_id += 1
                examples.append(example)

            return examples


if __name__ == "__main__":
    ds = NLPCC_DBQA_pairwise()
    print("first 10 train")
    for e in ds.get_train_examples()[:10]:
        print("{}\t{}\t{}\t{}\t{}".format(e.guid, e.qid, e.query, e.pos, e.neg))
    print("first 10 dev")
    for e in ds.get_dev_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print("first 10 test")
    for e in ds.get_test_examples()[:10]:
        print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
    print(ds)
