"""
Script that trains Tensorflow Multitask models on KINASE datasets.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import tempfile
import shutil
import numpy as np
import deepchem as dc
from KINASE_datasets import load_kinase

# Set numpy seed
np.random.seed(123)

###Load data###
shard_size = 2000
num_trials = 5
print("About to load KINASE data.")
KINASE_tasks, datasets, transformers = load_kinase(shard_size=shard_size)
train_dataset, valid_dataset, test_dataset = datasets

print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in validation set")
print(len(valid_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

all_results = []
for trial in range(num_trials):
  ###Create model###
  n_layers = 3
  nb_epoch = 50
  model = dc.models.TensorflowMultitaskRegressor(
      len(KINASE_tasks),
      train_dataset.get_data_shape()[0],
      layer_sizes=[1000] * n_layers,
      dropouts=[.25] * n_layers,
      weight_init_stddevs=[.02] * n_layers,
      bias_init_consts=[.5] * n_layers,
      learning_rate=.0003,
      penalty=.0001,
      penalty_type="l2",
      optimizer="adam",
      batch_size=100,
      verbosity="high")

  #Use R2 classification metric
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, task_averager=np.mean)

  print("Training model")
  model.fit(train_dataset, nb_epoch=nb_epoch, max_checkpoints_to_keep=1)

  print("Evaluating models")
  train_score, train_task_scores = model.evaluate(
      train_dataset, [metric], transformers, per_task_metrics=True)
  valid_score, valid_task_scores = model.evaluate(
      valid_dataset, [metric], transformers, per_task_metrics=True)
  test_score, test_task_scores = model.evaluate(
      test_dataset, [metric], transformers, per_task_metrics=True)

  all_results.append((train_score, train_task_scores, valid_score,
                      valid_task_scores, test_score, test_task_scores))

  print("Scores for trial %d" % trial)
  print("----------------------------------------------------------------")
  print("train_task_scores")
  print(train_task_scores)
  print("Mean Train score")
  print(train_score)
  print("valid_task_scores")
  print(valid_task_scores)
  print("Mean Validation score")
  print(valid_score)
  print("test_task_scores")
  print(test_task_scores)
  print("Mean Test score")
  print(test_score)

print("####################################################################")

for trial in range(num_trials):
  (train_score, train_task_scores, valid_score, valid_task_scores, test_score,
   test_task_scores) = all_results[trial]

  print("Scores for trial %d" % trial)
  print("----------------------------------------------------------------")
  print("train_task_scores")
  print(train_task_scores)
  print("Mean Train score")
  print(train_score)
  print("valid_task_scores")
  print(valid_task_scores)
  print("Mean Validation score")
  print(valid_score)
  print("test_task_scores")
  print(test_task_scores)
  print("Mean Test score")
  print(test_score)
