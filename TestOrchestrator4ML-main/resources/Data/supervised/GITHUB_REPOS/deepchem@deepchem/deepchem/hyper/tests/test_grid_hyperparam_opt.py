"""
Tests for hyperparam optimization.
"""
import unittest
import tempfile
import numpy as np
import deepchem as dc
import sklearn


class TestGridHyperparamOpt(unittest.TestCase):
  """
  Test grid hyperparameter optimization API.
  """

  def setUp(self):
    """Set up common resources."""

    def rf_model_builder(**model_params):
      rf_params = {k: v for (k, v) in model_params.items() if k != 'model_dir'}
      model_dir = model_params['model_dir']
      sklearn_model = sklearn.ensemble.RandomForestRegressor(**rf_params)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    self.rf_model_builder = rf_model_builder
    self.train_dataset = dc.data.NumpyDataset(
        X=np.random.rand(50, 5), y=np.random.rand(50, 1))
    self.valid_dataset = dc.data.NumpyDataset(
        X=np.random.rand(20, 5), y=np.random.rand(20, 1))

  def test_rf_hyperparam(self):
    """Test of hyperparam_opt with singletask RF ECFP regression API."""
    optimizer = dc.hyper.GridHyperparamOpt(self.rf_model_builder)
    params_dict = {"n_estimators": [10, 100]}
    transformers = []
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, self.train_dataset, self.valid_dataset, metric,
        transformers)
    valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                      transformers)

    assert valid_score["pearson_r2_score"] == max(all_results.values())
    assert valid_score["pearson_r2_score"] > 0

  def test_rf_hyperparam_min(self):
    """Test of hyperparam_opt with singletask RF ECFP regression API."""
    optimizer = dc.hyper.GridHyperparamOpt(self.rf_model_builder)
    params_dict = {"n_estimators": [10, 100]}
    transformers = []
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        self.train_dataset,
        self.valid_dataset,
        metric,
        transformers,
        use_max=False)
    valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                      transformers)

    assert valid_score["pearson_r2_score"] == min(all_results.values())
    assert valid_score["pearson_r2_score"] > 0

  def test_rf_with_logdir(self):
    """Test that using a logdir can work correctly."""
    optimizer = dc.hyper.GridHyperparamOpt(self.rf_model_builder)
    params_dict = {"n_estimators": [10, 5]}
    transformers = []
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    with tempfile.TemporaryDirectory() as tmpdirname:
      best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
          params_dict,
          self.train_dataset,
          self.valid_dataset,
          metric,
          transformers,
          logdir=tmpdirname)
    valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                      transformers)
    assert valid_score["pearson_r2_score"] == max(all_results.values())
    assert valid_score["pearson_r2_score"] > 0

  def test_multitask_example(self):
    """Test a simple example of optimizing a multitask model with a grid search."""
    # Generate dummy dataset
    np.random.seed(123)
    train_dataset = dc.data.NumpyDataset(
        np.random.rand(10, 3), np.zeros((10, 2)), np.ones((10, 2)),
        np.arange(10))
    valid_dataset = dc.data.NumpyDataset(
        np.random.rand(5, 3), np.zeros((5, 2)), np.ones((5, 2)), np.arange(5))

    optimizer = dc.hyper.GridHyperparamOpt(
        lambda **params: dc.models.MultitaskRegressor(n_tasks=2,
                                                      n_features=3, dropouts=[0.],
                                                      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
                                                      learning_rate=0.003, **params))

    params_dict = {"batch_size": [10, 20]}
    transformers = []
    metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean)

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        train_dataset,
        valid_dataset,
        metric,
        transformers,
        use_max=False)

    valid_score = best_model.evaluate(valid_dataset, [metric])
    assert valid_score["mean-mean_squared_error"] == min(all_results.values())
    assert valid_score["mean-mean_squared_error"] > 0

  def test_multitask_example_multiple_params(self):
    """Test a simple example of optimizing a multitask model with a grid search with multiple parameters to optimize."""
    # Generate dummy dataset
    np.random.seed(123)
    train_dataset = dc.data.NumpyDataset(
        np.random.rand(10, 3), np.zeros((10, 2)), np.ones((10, 2)),
        np.arange(10))
    valid_dataset = dc.data.NumpyDataset(
        np.random.rand(5, 3), np.zeros((5, 2)), np.ones((5, 2)), np.arange(5))

    optimizer = dc.hyper.GridHyperparamOpt(
        lambda **params: dc.models.MultitaskRegressor(
            n_tasks=2,
            n_features=3,
            dropouts=[0.],
            weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
            **params))

    params_dict = {"learning_rate": [0.003, 0.03], "batch_size": [10, 50]}
    # These are per-example multiplier
    transformers = []
    metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean)

    with tempfile.TemporaryDirectory() as tmpdirname:
      best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
          params_dict,
          train_dataset,
          valid_dataset,
          metric,
          transformers,
          logdir=tmpdirname,
          use_max=False)
      valid_score = best_model.evaluate(valid_dataset, [metric])
    # Test that 2 parameters were optimized
    for hp_str in all_results.keys():
      # Recall that the key is a string of the form _batch_size_39_learning_rate_0.01 for example
      assert "batch_size" in hp_str
      assert "learning_rate" in hp_str

    assert valid_score["mean-mean_squared_error"] == min(all_results.values())
    assert valid_score["mean-mean_squared_error"] > 0
