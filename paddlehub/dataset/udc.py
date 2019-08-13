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
import pandas as pd
import csv

from paddlehub.dataset import InputExample, HubDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/udc.tar.gz"


class UDC(HubDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)
    """
    def __init__(self):
        self.dataset_dir = os.path.join(DATA_HOME, "udc")
        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.txt")
        self.train_examples = self._read_txt(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.txt")
        # Dev and test data sets are too big (500,000)
        # To speed up dev/test stage, we  only hold out 5000 samples.
        self.dev_examples = self._read_txt(self.dev_file, limit=5000)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.txt")
        self.test_examples = self._read_txt(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_txt(self, input_file, quotechar=None, limit=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = []
            seq_id = 0
            for num, line in enumerate(reader):
                example = InputExample(guid=seq_id,
                                       label=line[0],
                                       text_a=line[1:-1],
                                       text_b=line[-1])
                seq_id += 1
                examples.append(example)
                if limit and num + 1 >= limit:
                    break
            return examples


if __name__ == "__main__":
    ds = UDC()
    #     for e in ds.get_train_examples()[:3]:
    #         print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))

    total_len = 0
    max_len = 0
    total_num = over_num = 0
    overlen = []
    for e in ds.get_dev_examples():
        length = len(" ".join(e.text_a).split()) + len(
            e.text_b.split()) if e.text_b else len(e.text_a.split())
        total_len += length
        if length > max_len:
            max_len = length
        total_num += 1
        if length > 256:
            over_num += 1
            overstr = ("\ntext_a: " + " ".join(e.text_a) + "\ntext_b:" +
                       e.text_b) if e.text_b else " ".join(e.text_a)
            overlen.append(overstr)
    avg = total_len / total_num
    for o in overlen[:2]:
        print("The data length>256:{}".format(o))
    print(
        "The total number: {}\nThe avrage length: {}\nthe max length: {}\nthe number of data length > 256:  {}"
        .format(total_num, avg, max_len, over_num))
