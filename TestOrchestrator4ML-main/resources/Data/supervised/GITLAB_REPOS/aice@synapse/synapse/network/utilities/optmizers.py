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
from tensorflow.keras import optimizers as keras_optimizers
# endregion : Imports


class Optimizers:
    AdaDelta = keras_optimizers.Adadelta
    AdaGrad = keras_optimizers.Adagrad
    Adam = keras_optimizers.Adam
    AdaMax = keras_optimizers.Adamax
    Ftrl = keras_optimizers.Ftrl
    NAdam = keras_optimizers.Nadam
    RMSProp = keras_optimizers.RMSprop
    SGD = keras_optimizers.SGD
