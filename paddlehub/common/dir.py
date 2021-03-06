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

import os

# TODO: Change dir.py's filename, this naming rule is not qualified
USER_HOME = os.path.expanduser('~')
HUB_HOME = os.path.join(USER_HOME, ".paddlehub")
MODULE_HOME = os.path.join(HUB_HOME, "modules")
CACHE_HOME = os.path.join(HUB_HOME, "cache")
DATA_HOME = os.path.join(HUB_HOME, "dataset")
CONF_HOME = os.path.join(HUB_HOME, "conf")
