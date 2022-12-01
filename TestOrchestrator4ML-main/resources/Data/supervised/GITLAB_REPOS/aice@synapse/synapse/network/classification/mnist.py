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
from synapse.network import Network
from synapse.network.utilities import Padding
from synapse.network.utilities.layers import Layers
from synapse.network.utilities.activations import Activations
from synapse.network.utilities.initialiers import Initializers
# endregion : Imports


class Mnist(Network):
    def __init__(self, config, num_output, *args, **kwargs):
        super(Mnist, self).__init__(config, *args, **kwargs)
        self.conv1 = Layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   data_format=self.data_format,
                                   padding=Padding.Valid,
                                   kernel_initializer=Initializers.he_uniform,
                                   activation=Activations.ReLU)
        self.flatten = Layers.Flatten(data_format=self.data_format)
        self.dense1 = Layers.Dense(units=128,
                                   activation=Activations.ReLU,
                                   kernel_initializer=Initializers.he_uniform)
        self.dense2 = Layers.Dense(units=num_output)
        self.softmax = Layers.Softmax()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x

