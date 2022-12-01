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
import os
from datetime import datetime
from synapse.Datasets import PublicDataset
from synapse.network.classification.mnist import Mnist
from synapse.config.classification import ClassificationConfig
from synapse.data_handler.dataset_iterator import DatasetIterator
from synapse.network.utilities import Losses, Optimizers, Metrics, CallBacks
from synapse.config import TrainVerbose, DataFormat
# endregion

# Main Config
Config = ClassificationConfig
name = 'MNIST'
log_dir = "logs"
unique_sig = os.path.join(name, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_path = os.path.join(log_dir, unique_sig, 'tensorboard', )
checkpoint_path = os.path.join(log_dir, unique_sig, 'checkpoints', "%s-epoch.{epoch:02d}.hdf5" % name)

# Project
Config.PROJECT_CONFIG.NAME = name
Config.PROJECT_CONFIG.BATCH_SIZE = 64
Config.PROJECT_CONFIG.MAX_EPOCHS = 10
Config.PROJECT_CONFIG.SAVE_STEPS = 1000
Config.PROJECT_CONFIG.TEST_STEPS = 1000
Config.PROJECT_CONFIG.SUMMARY_STEPS = 1000
Config.PROJECT_CONFIG.TRAINING_VERBOSE = TrainVerbose.ProgressBar
Config.PROJECT_CONFIG.LOG_DIR = log_dir
Config.PROJECT_CONFIG.TENSORBOARD_DIR = tensorboard_path
Config.PROJECT_CONFIG.CHECKPOINT_PATH = checkpoint_path
Config.PROJECT_CONFIG.CALLBACKS = [
    CallBacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    ),
    CallBacks.ProgbarLogger(),
    CallBacks.Tensorboard(log_dir=tensorboard_path),
    CallBacks.TerminateOnNan()
]


# Data
Config.DATA_CONFIG.DATASET = PublicDataset(
    name='MNIST',
    download_dir='./data'
)
Config.DATA_CONFIG.IMAGE_SIZE = [28, 28]
Config.DATA_CONFIG.CHANNELS = 1
Config.DATA_CONFIG.DATA_FORMAT = DataFormat.ChannelsLast
Config.DATA_CONFIG.ITERATOR = DatasetIterator(
    dataset=Config.DATA_CONFIG.DATASET
)
# Model
Config.MODEL_CONFIG.NETWORK = Mnist
Config.MODEL_CONFIG.LOSS = Losses.SparseCategoricalCrossEntropy
Config.MODEL_CONFIG.LOSS_KWARGS = {
    'from_logits': True,
}
Config.MODEL_CONFIG.OPTIMIZER = Optimizers.Adam
Config.MODEL_CONFIG.OPTIMIZER_KWARGS = {
    'learning_rate': 0.01,
}
Config.MODEL_CONFIG.METRICS = [
    Metrics.SparseCategoricalCrossEntropy(name='accuracy'),
]

Config.MODEL_CONFIG.COMPILE_KWARGS = {
    'optimizer': Optimizers.Adam(learning_rate=0.00001),
    'loss': Config.MODEL_CONFIG.LOSS(**Config.MODEL_CONFIG.LOSS_KWARGS),
    'metrics': ['accuracy', 'mse'],
    'loss_weights': None,
    'sample_weight_mode': None,
    'weighted_metrics': None,
    'target_tensors': None,
    'distribute': None,
}



