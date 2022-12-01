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
from tensorflow.keras import callbacks as keras_callbacks

# endregion : Imports


class CallBacks:
    """
    Ref: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
    """
    BaseLogger = keras_callbacks.BaseLogger
    CSVLogger = keras_callbacks.CSVLogger
    EarlyStopping = keras_callbacks.EarlyStopping
    History = keras_callbacks.History
    LambdaCallBack = keras_callbacks.LambdaCallback
    LearningRateScheduler = keras_callbacks.LearningRateScheduler
    ModelCheckpoint = keras_callbacks.ModelCheckpoint
    ProgbarLogger = keras_callbacks.ProgbarLogger
    ReduceROnPlateau = keras_callbacks.ReduceLROnPlateau
    RemoteMonitor = keras_callbacks.RemoteMonitor
    Tensorboard = keras_callbacks.TensorBoard
    TerminateOnNan = keras_callbacks.TerminateOnNaN
