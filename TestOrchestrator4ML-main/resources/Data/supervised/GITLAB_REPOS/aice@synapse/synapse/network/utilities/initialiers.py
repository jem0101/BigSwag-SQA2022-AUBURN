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


class Initializers:
    """
    Ref. : https://www.tensorflow.org/api_docs/python/tf/keras/initializers
    """
    Constant = 'constant'
    GlorotNormal = 'glorot_normal'
    GlorotUniform = 'glorot_uniform'
    Identity = 'identity'
    Ones = 'ones'
    Orthogonal = 'orthogonal'
    RandomNormal = 'random_normal'
    RandomUniform = 'random_uniform'
    TrancatedNormal = 'trancated_normal'
    VarianceScaling = 'variance_scaling'
    Zeros = 'zeros'
    he_normal = 'he_normal'
    he_uniform = 'he_uniform'
    lecun_normal = 'lecun_normal'
    lecun_uniform = 'lecun_uniform'
