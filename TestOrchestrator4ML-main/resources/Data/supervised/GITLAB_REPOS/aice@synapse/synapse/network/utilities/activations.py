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
from tensorflow.keras import activations as keras_activations
# endregion : Imports


class Activations:
    """
    Ref. : https://www.tensorflow.org/api_docs/python/tf/keras/initializers
    """
    ELU = keras_activations.elu
    Exponential = keras_activations.exponential
    HardSigmoid = keras_activations.hard_sigmoid
    Linear = keras_activations.linear
    ReLU = keras_activations.relu
    SeLU = keras_activations.selu
    Sigmoid = keras_activations.sigmoid
    SoftMax = keras_activations.softmax
    SoftPlus = keras_activations.softplus
    SoftSign = keras_activations.softsign
    Tanh = keras_activations.tanh

