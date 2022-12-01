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
# endregion : Imports


# region : Enums
class DataFormat:
    ChannelsFirst = 'channels_first'
    ChannelsLast = 'channels_last'


class ModelType(Enum):
    Classification = 0
    ObjectDetection = 1
    Segmentation = 2


class DataType(Enum):
    PlainText = 0
    CSV = 1
    JSON = 2
    NumpyPickle = 3
    TFRecord = 4


class DatasetSplit(Enum):
    train = 0
    validation = 1
    test = 2


class TFLogLevel(Enum):
    All = 0
    WarningAndError = 1
    ErrorOnly = 2
    NoLog = 3


class TrainVerbose(Enum):
    Silent = 0
    ProgressBar = 1
    OneLinePerEpoch = 2
# endregion : Enums


