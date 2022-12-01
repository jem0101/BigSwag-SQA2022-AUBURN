"""
Copyright (C) 2019  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# region : Imports
from . import BaseIterator
from synapse.Datasets import PublicDataset

import tensorflow as tf
# endregion : Imports


class DatasetIterator(BaseIterator):
    """
    Dataset Iterator class. It takes dataset and necessary information to provide a dataset iterator.

    :param Dataset: Dataset as dict format with subset {train, test ..} as key and tf.Dataset as value.
    """
    def __init__(self, dataset, *args, **kwargs):
        assert dataset.is_initialized, "Dataset Iterator init failed. Dataset is not initialized..."
        super(DatasetIterator, self).__init__(dataset, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        sub_dataset = self.dataset(*args, **kwargs)
        assert sub_dataset, "Subset name not in dataset..."

        return iter(sub_dataset)




