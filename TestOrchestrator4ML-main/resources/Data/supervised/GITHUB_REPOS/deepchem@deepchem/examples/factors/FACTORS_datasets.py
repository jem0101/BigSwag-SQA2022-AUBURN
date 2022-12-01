"""
FACTORS dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import time
import numpy as np
import deepchem as dc
from factors_features import factors_descriptors 

def remove_missing_entries(dataset):
  """Remove missing entries.

  Some of the datasets have missing entries that sneak in as zero'd out
  feature vectors. Get rid of them.
  """
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    print("Shard %d has %d missing entries."
        % (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)

def get_transformers(train_dataset):
  """Get transformers applied to datasets."""
  transformers = []
  return transformers

def gen_factors(FACTORS_tasks, raw_train_dir, train_dir, valid_dir, test_dir,
                shard_size=10000):
  """Load Factor datasets."""
  train_files = ("FACTORS_training_disguised_combined_full.csv.gz")
  valid_files = ("FACTORS_test1_disguised_combined_full.csv.gz")
  test_files = ("FACTORS_test2_disguised_combined_full.csv.gz")

  # Featurize FACTORS dataset
  print("About to featurize FACTORS dataset.")
  featurizer = dc.feat.UserDefinedFeaturizer(factors_descriptors)

  loader = dc.data.UserCSVLoader(
      tasks=FACTORS_tasks, id_field="Molecule", featurizer=featurizer)

  train_datasets, valid_datasets, test_datasets = [], [], []
  print("Featurizing train datasets")
  train_dataset = loader.featurize(train_files, shard_size=shard_size)

  print("Featurizing valid datasets")
  valid_dataset = loader.featurize(valid_files, shard_size=shard_size)

  print("Featurizing test datasets")
  test_dataset = loader.featurize(test_files, shard_size=shard_size)

  print("Remove missing entries from datasets.")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  print("Transforming datasets with transformers.")
  transformers = get_transformers(train_dataset)
  raw_train_dataset = train_dataset

  for transformer in transformers:
    print("Performing transformations with %s"
          % transformer.__class__.__name__)
    print("Transforming datasets")
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  print("Shuffling order of train dataset.")
  train_dataset.sparse_shuffle()

  print("Moving directories")
  raw_train_dataset.move(raw_train_dir)
  train_dataset.move(train_dir)
  valid_dataset.move(valid_dir)
  test_dataset.move(test_dir)
  
  return (raw_train_dataset, train_dataset, valid_dataset, test_dataset)

def load_factors(shard_size):
  """Loads factors datasets. Generates if not stored already."""
  FACTORS_tasks = (['T_0000%d' % i for i in range(1, 10)]
                   + ['T_000%d' % i for i in range(10, 13)])

  current_dir = os.path.dirname(os.path.realpath(__file__))
  raw_train_dir = os.path.join(current_dir, "raw_train_dir")
  train_dir = os.path.join(current_dir, "train_dir") 
  valid_dir = os.path.join(current_dir, "valid_dir") 
  test_dir = os.path.join(current_dir, "test_dir") 

  if (os.path.exists(raw_train_dir) and
      os.path.exists(train_dir) and
      os.path.exists(valid_dir) and
      os.path.exists(test_dir)):
    print("Reloading existing datasets")
    raw_train_dataset = dc.data.DiskDataset(raw_train_dir)
    train_dataset = dc.data.DiskDataset(train_dir)
    valid_dataset = dc.data.DiskDataset(valid_dir)
    test_dataset = dc.data.DiskDataset(test_dir)
  else:
    print("Featurizing datasets")
    (raw_train_dataset, train_dataset, valid_dataset, test_dataset) = \
      gen_factors(FACTORS_tasks, raw_train_dir, train_dir, valid_dir, test_dir,
                  shard_size=shard_size)

  transformers = get_transformers(raw_train_dataset)
  return FACTORS_tasks, (train_dataset, valid_dataset, test_dataset), transformers
