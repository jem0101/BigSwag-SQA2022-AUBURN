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
import os, shutil
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger, TerminateOnNaN

# endregion: Imports


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.dataset_iterator = None
        self.model = None
        self.loss_ops = None
        self.optimizer = None
        self.metrics_ops = None
        self.max_steps = 0
        self.max_epoch = 0

        self.name = None
        self.batch_size = 0
        self.save_steps = 0
        self.test_steps = 0
        self.summary_steps = 0
        self.verbose = 0
        self.callbacks = None
        self.logdir = None
        self.tensorboard_dir = None
        self.checkpoint_path = None

    def prepare_dataset(self):
        print("Preparing Dataset .........")
        self.dataset = self.config.DATA_CONFIG.DATASET
        print("Preparing Dataset Iterator .........")
        self.dataset_iterator = self.config.DATA_CONFIG.ITERATOR

    def prepare_model(self):
        print("Preparing Model .........")
        self.model = self.config.MODEL_CONFIG.NETWORK(self.config, self.config.DATA_CONFIG.DATASET.num_classes)
        input_shape = tuple([None] + self.config.DATA_CONFIG.IMAGE_SIZE + [self.config.DATA_CONFIG.CHANNELS])
        self.model.build(input_shape=input_shape)
        print("Preparing Loss .........")
        self.loss_ops = self.config.MODEL_CONFIG.LOSS(**self.config.MODEL_CONFIG.LOSS_KWARGS)
        print("Preparing Optimizer .........")
        self.optimizer = self.config.MODEL_CONFIG.OPTIMIZER(**self.config.MODEL_CONFIG.OPTIMIZER_KWARGS)
        print("Preparing Training Metrics ..........")
        self.metrics_ops = self.config.MODEL_CONFIG.METRICS
        print("Compiling Model ..........")
        self.model.compile(**self.config.MODEL_CONFIG.COMPILE_KWARGS)
        print("Building Model ...........")

    def prepare_training(self):
        self.name = self.config.PROJECT_CONFIG.NAME
        self.batch_size = self.config.PROJECT_CONFIG.BATCH_SIZE
        self.max_epoch = self.config.PROJECT_CONFIG.MAX_EPOCHS
        self.save_steps = self.config.PROJECT_CONFIG.SAVE_STEPS
        self.test_steps = self.config.PROJECT_CONFIG.TEST_STEPS
        self.summary_steps = self.config.PROJECT_CONFIG.SUMMARY_STEPS
        self.verbose = self.config.PROJECT_CONFIG.TRAINING_VERBOSE
        if self.config.PROJECT_CONFIG.LOG_DIR:
            self.logdir = self.config.PROJECT_CONFIG.LOG_DIR
            self.tensorboard_dir = self.config.PROJECT_CONFIG.TENSORBOARD_DIR
            self.checkpoint_path = self.config.PROJECT_CONFIG.CHECKPOINT_PATH
            self.callbacks = self.config.PROJECT_CONFIG.CALLBACKS
            os.makedirs(os.path.dirname(self.tensorboard_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)


    def prepare_network(self):
        self.prepare_dataset()
        self.prepare_model()
        self.prepare_training()
