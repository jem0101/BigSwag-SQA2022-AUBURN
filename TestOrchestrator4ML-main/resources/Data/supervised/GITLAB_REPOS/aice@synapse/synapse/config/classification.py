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
from easydict import EasyDict

from . import *
from synapse.network.utilities.losses import Losses
from synapse.network.utilities.metrics import Metrics
from synapse.network.utilities.optmizers import Optimizers
from synapse.network.utilities.initialiers import Initializers

name = 'MNIST'
log_dir = "logs"


# endregion : Imports
# region : Classes
class ClassificationConfig:
    DATA_CONFIG = EasyDict(
        {
            'DATASET': None,
            'IMAGE_SIZE': [28, 28],
            'CHANNELS': 3,
            'DATA_FORMAT': DataFormat.ChannelsFirst,
            'ITERATOR': None
        }
    )
    PROJECT_CONFIG = EasyDict(
        {
            'NAME': 'untitled',
            'BATCH_SIZE': 64,
            'MAX_EPOCHS': 10,
            'SAVE_STEPS': 1000,
            'TEST_STEPS': 1000,
            'SUMMARY_STEPS': 1000,
            'TRAINING_VERBOSE': TrainVerbose.ProgressBar,
            'LOG_DIR': None,
            'TENSORBOARD_DIR': None,
            'CHECKPOINT_PATH': None,
            'CALLBACKS': None
        }
    )
    MODEL_CONFIG = EasyDict(
        {
            'NETWORK': None,
            'OPTIMIZER': Optimizers.SGD,
            'OPTIMIZER_KWARGS': None,
            'LOSS': Losses.SparseCategoricalCrossEntropy,
            'LOSS_KWARGS': None,
            'KERNEL_INITIALIZER': Initializers.GlorotUniform,
            'KERNEL_KWARGS': {
                'seed': None
            },
            'METRICS': {
                'loss': Metrics.Mean,
                'accuracy': Metrics.SparseCategoricalCrossEntropy
            },
            'COMPILE_ARGS': {
                'optimizer': None,
                'loss': None,
                'metrics': None,
                'loss_weights': None,
                'sample_weight_mode': None,
                'weighted_metrics': None,
                'target_tensors': None,
                'distribute=None': None
            }
        }
    )
