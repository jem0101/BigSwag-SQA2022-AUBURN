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
from tensorflow.keras import regularizers as keras_regularizers
# endregion : Imports


class Regularizers:
    L1L2 = keras_regularizers.L1L2
    Regularizer = keras_regularizers.Regularizer
    l1 = keras_regularizers.l1
    l1_l2 = keras_regularizers.l1_l2
    l2 = keras_regularizers.l2
