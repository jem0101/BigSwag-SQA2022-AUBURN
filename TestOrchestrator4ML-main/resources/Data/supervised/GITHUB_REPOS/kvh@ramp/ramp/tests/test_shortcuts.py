import os
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal
from sklearn import linear_model

from ramp.estimators.base import Probabilities
from ramp.features.base import F, Map
from ramp.features.trained import Predictions
from ramp.model_definition import ModelDefinition
from ramp.shortcuts import cross_validate, cv_factory, param_search
from ramp.tests.test_features import make_data
from ramp.tests.test_modeling import DummyEstimator


class CloneableEstimator(DummyEstimator):

    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.params = kwargs

    def get_params(self, deep=False):
        return self.params


class TestShortcuts(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_cross_validate(self):
        folds = 3
        estimator = linear_model.LinearRegression()
        cvresult = cross_validate(self.data, folds=folds,
                                          features = [F(10), F('a')],
                                          target = F('b'),
                                          estimator = estimator)
        self.assertEqual(len(cvresult.results), folds)
        self.assertEqual(cvresult.model_def.estimator.base_estimator_, estimator)

    def test_cross_validate_factory(self):
        folds = 3
        n_models = 2
        cvcresult = cv_factory(self.data,
                              folds=folds,
                              features=[[F(10), F('a')]],
                              target=[F('b'), F('a')],
                              estimator=[linear_model.LinearRegression()])
        self.assertEqual(len(cvcresult.cvresults), n_models)

    def test_param_search(self):
        grid = param_search(CloneableEstimator(),
                            {'param1':range(3), 'param2':['a','b']})
        self.assertEqual(len(grid), 6)


if __name__ == '__main__':
    unittest.main()


