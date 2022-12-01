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
from . import BaseTrainer
from synapse.data_handler.dataset_iterator import DatasetIterator
# endregion : Imports


class Trainer(BaseTrainer):
    def build_data_feeder(self, *args, **kwargs):
        dataset = self.config.dataset
        batch_size = self.config.ProjectConfig.batch_size
        shuffle = self.config.ProjectConfig.shuffle
        prefetch = self.config.ProjectConfig.prefetch
        train_ds, val_ts = DatasetIterator(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           prefetch=prefetch)

    def build_saver(self, *args, **kwargs):
        pass

    def build_summary(self, *args, **kwargs):
        pass

    def build_graph(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def __init__(self, config):
        super(Trainer, self).__init__(config)