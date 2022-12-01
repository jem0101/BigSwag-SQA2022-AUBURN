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
# region : Import
from . import BaseTrainer
import tensorflow as tf
import os, math
from synapse.config import TrainVerbose, DataFormat
# endregion : Import


class SimpleTrainer(BaseTrainer):
    def __init__(self, config):
        super(SimpleTrainer, self).__init__(config)
        self.prepare_network()

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            cost = self.loss_ops(labels, predictions)

        gradients = tape.gradient(cost, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.metrics_ops(cost, labels, predictions)

    # @tf.function
    def train(self,
              batch_size=None,
              epochs=1,
              max_steps=None,
              verbose=TrainVerbose.ProgressBar,
              callbacks=None,
              validation_split=0.0,
              validation_data=None,
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_steps=None,
              validation_freq=1,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              **kwargs):

        if batch_size:
            self.batch_size = batch_size
        if max_steps:
            self.max_epoch = max(self.max_steps // self.dataset.data_count('train'), 1) if self.max_steps else epochs
        else:
            self.max_epoch = epochs
        if callbacks:
            self.callbacks = callbacks

        train_data = self.dataset(subset='train',
                                  shuffle=shuffle,
                                  data_format=DataFormat.ChannelsLast,
                                  batch=self.batch_size)

        print(self.model.summary())
        print()
        print("Ready for training ...........")
        print("--> Max Epochs : ", self.max_epoch)
        print("--> 1 Epochs : ", self.dataset.data_count('train'))
        print("--> Batch Size : ", self.batch_size)
        print("--> Steps/Epochs : ", math.ceil(self.dataset.data_count('train') / self.batch_size))
        print("--> Tensorboard Dir : ", "N/A" if not self.tensorboard_dir else self.tensorboard_dir)
        print("--> Checkpoint Dir : ", "N/A" if not self.checkpoint_path else os.path.dirname(self.checkpoint_path))
        print()

        self.model.fit(train_data,
                       epochs=self.max_epoch,
                       verbose=verbose.value,
                       callbacks=self.callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       class_weight=class_weight,
                       sample_weight=sample_weight,
                       initial_epoch=initial_epoch,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_freq=validation_freq,
                       max_queue_size=max_queue_size,
                       workers=workers,
                       use_multiprocessing=use_multiprocessing,
                       **kwargs)



