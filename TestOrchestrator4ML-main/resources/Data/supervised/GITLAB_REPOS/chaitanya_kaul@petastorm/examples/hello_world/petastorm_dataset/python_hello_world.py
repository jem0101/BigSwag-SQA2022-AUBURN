#  Copyright (c) 2017-2018 Uber Technologies, Inc.
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

"""Minimal example of how to read samples from a dataset generated by `generate_hello_world_dataset.py`
using plain Python"""

from __future__ import print_function

from petastorm import make_reader


def python_hello_world(dataset_url='file:///tmp/hello_world_dataset'):
    with make_reader(dataset_url) as reader:
        # Pure python
        for sample in reader:
            print(sample.id)
            # plt.imshow(sample.image1)


if __name__ == '__main__':
    python_hello_world()
