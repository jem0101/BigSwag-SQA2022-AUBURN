"""
Copyright (C) 2020  Syed Hasibur Rahman

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
from abc import abstractmethod
from synapse.config import DataFormat
from tensorflow.keras import Model
# end region : Imports


class Network(Model):
    def __init__(self, config, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.config = config
        self.data_format = config.DATA_CONFIG.DATA_FORMAT

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError


