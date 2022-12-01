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
from enum import Enum

from .layers import Layers
from .losses import Losses
from .metrics import Metrics
from .callbacks import CallBacks
from .optmizers import Optimizers
from .activations import Activations
from .initialiers import Initializers

# endregion : Imports


class Padding:
    Same = 'SAME'
    Valid = 'VALID'
