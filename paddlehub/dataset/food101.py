#coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import ImageClassificationDataset


class Food101Dataset(ImageClassificationDataset):
    def __init__(self):
        super(Food101Dataset, self).__init__()
        dataset_path = os.path.join(hub.common.dir.DATA_HOME, "food-101",
                                    "images")
        self.base_path = self._download_dataset(
            dataset_path=dataset_path,
            url="https://paddlehub-dataset.bj.bcebos.com/Food101.tar.gz")
        self.train_list_file = "train_list.txt"
        self.test_list_file = "test_list.txt"
        self.validate_list_file = "validate_list.txt"
        self.label_list_file = "label_list.txt"
        self.num_labels = 101
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_examples = self._read_data(self.base_path + "/" +
                                              self.train_list_file)
        self.num_examples["train"] = len(self.train_examples)

    def _load_dev_examples(self):
        self.dev_examples = self._read_data(self.base_path + "/" +
                                            self.validate_list_file)
        self.num_examples["dev"] = len(self.dev_examples)

    def _load_test_examples(self):
        self.test_examples = self._read_data(self.base_path + "/" +
                                             self.test_list_file)
        self.num_examples["test"] = len(self.test_examples)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def _read_data(self, data_path):
        data = []
        with open(data_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                items = line.split(" ")
                if len(items) > 2:
                    image_path = " ".join(items[0:-1])
                else:
                    image_path = items[0]
                if not os.path.isabs(image_path):
                    if self.base_path is not None:
                        image_path = os.path.join(self.base_path, image_path)
                label = items[-1]
                data.append((image_path, items[-1]))

        return data
