"""This example implements RF experiments from https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00290"""
import sys
import os
import deepchem
import deepchem as dc
import tempfile, shutil
from bace_datasets import load_bace
from deepchem.hyper import HyperparamOpt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.sklearn_models import SklearnModel
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator


def bace_rf_model(mode="classification", split="20-80"):
  """Train random forests on BACE dataset."""
  (bace_tasks, (train, valid, test, crystal), transformers) = load_bace(
      mode=mode, transform=False, split=split)

  if mode == "regression":
    r2_metric = Metric(metrics.r2_score)
    rms_metric = Metric(metrics.rms_score)
    mae_metric = Metric(metrics.mae_score)
    all_metrics = [r2_metric, rms_metric, mae_metric]
    metric = r2_metric
    model_class = RandomForestRegressor

    def rf_model_builder(model_params, model_dir):
      sklearn_model = RandomForestRegressor(**model_params)
      return SklearnModel(sklearn_model, model_dir)
  elif mode == "classification":
    roc_auc_metric = Metric(metrics.roc_auc_score)
    accuracy_metric = Metric(metrics.accuracy_score)
    mcc_metric = Metric(metrics.matthews_corrcoef)
    # Note sensitivity = recall
    recall_metric = Metric(metrics.recall_score)
    model_class = RandomForestClassifier
    all_metrics = [accuracy_metric, mcc_metric, recall_metric, roc_auc_metric]
    metric = roc_auc_metric

    def rf_model_builder(model_params, model_dir):
      sklearn_model = RandomForestClassifier(**model_params)
      return SklearnModel(sklearn_model, model_dir)
  else:
    raise ValueError("Invalid mode %s" % mode)

  params_dict = {
      "n_estimators": [10, 100],
      "max_features": ["auto", "sqrt", "log2", None],
  }

  optimizer = HyperparamOpt(rf_model_builder)
  best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
      params_dict, train, valid, transformers, metric=metric)

  if len(train) > 0:
    rf_train_evaluator = Evaluator(best_rf, train, transformers)
    csv_out = "rf_%s_%s_train.csv" % (mode, split)
    stats_out = "rf_%s_%s_train_stats.txt" % (mode, split)
    rf_train_score = rf_train_evaluator.compute_model_performance(
        all_metrics, csv_out=csv_out, stats_out=stats_out)
    print("RF Train set scores: %s" % (str(rf_train_score)))

  if len(valid) > 0:
    rf_valid_evaluator = Evaluator(best_rf, valid, transformers)
    csv_out = "rf_%s_%s_valid.csv" % (mode, split)
    stats_out = "rf_%s_%s_valid_stats.txt" % (mode, split)
    rf_valid_score = rf_valid_evaluator.compute_model_performance(
        all_metrics, csv_out=csv_out, stats_out=stats_out)
    print("RF Valid set scores: %s" % (str(rf_valid_score)))

  if len(test) > 0:
    rf_test_evaluator = Evaluator(best_rf, test, transformers)
    csv_out = "rf_%s_%s_test.csv" % (mode, split)
    stats_out = "rf_%s_%s_test_stats.txt" % (mode, split)
    rf_test_score = rf_test_evaluator.compute_model_performance(
        all_metrics, csv_out=csv_out, stats_out=stats_out)
    print("RF Test set: %s" % (str(rf_test_score)))

  if len(crystal) > 0:
    rf_crystal_evaluator = Evaluator(best_rf, crystal, transformers)
    csv_out = "rf_%s_%s_crystal.csv" % (mode, split)
    stats_out = "rf_%s_%s_crystal_stats.txt" % (mode, split)
    rf_crystal_score = rf_crystal_evaluator.compute_model_performance(
        all_metrics, csv_out=csv_out, stats_out=stats_out)
    print("RF Crystal set: %s" % (str(rf_crystal_score)))


if __name__ == "__main__":
  print("Classifier RF 20-80:")
  print("--------------------------------")
  bace_rf_model(mode="classification", split="20-80")
  print("Classifier RF 80-20:")
  print("--------------------------------")
  bace_rf_model(mode="classification", split="80-20")

  print("Regressor RF 20-80:")
  print("--------------------------------")
  bace_rf_model(mode="regression", split="20-80")
  print("Regressor RF 80-20:")
  print("--------------------------------")
  bace_rf_model(mode="regression", split="80-20")
