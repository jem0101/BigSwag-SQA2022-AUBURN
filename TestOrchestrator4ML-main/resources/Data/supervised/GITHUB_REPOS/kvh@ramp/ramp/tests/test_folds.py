import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index

from ramp.features.base import F, Map
from ramp.folds import *
from ramp.utils import *


class TestFolds(unittest.TestCase):

    def test_watertight_folds(self):
        n   = 10000
        n_u = 1000
        r   = 4
        n_folds = 10
        df = pd.DataFrame({'a': np.arange(n),
                           'u': np.random.randint(1, n_u, n)})

        wt_folds = WatertightFolds(n_folds, df, 'u', seed=1)
        folds = list(wt_folds)
        self.assertEqual(len(folds), n_folds)
        te = pd.Index([])
        u_sofar = set()
        for train, test in folds:
            self.assertFalse(train & test)
            self.assertFalse(te & test)
            self.assertEqual(len(u_sofar.intersection(df.loc[test])), 0)
            te = te | test
            u_sofar = u_sofar.union(df.loc[test]['u'])
        # ensure all instances were used in test
        self.assertEqual(len(set(df['u'])), len(u_sofar))
        self.assertEqual(len(te), n)

    def test_balanced_folds(self):
        n = 100000
        r = 4
        n_folds = 4
        df = pd.DataFrame({'a':np.arange(n),
                           'y':np.hstack([np.ones(n/r), np.zeros(n/r * (r -1))])})
        balanced_folds = BalancedFolds(n_folds, F('y'), df, seed=1)
        folds = list(balanced_folds)
        self.assertEqual(len(folds), n_folds)
        te = pd.Index([])
        for train, test in folds:
            self.assertFalse(train & test)
            self.assertFalse(te & test)
            te = te | test
            train_y = df.y[train]
            test_y = df.y[test]
            # ensure postive ratios are correct
            pos_ratio = sum(train_y) / float(len(train_y))
            self.assertAlmostEqual(pos_ratio, 1. / r)
            pos_ratio = sum(test_y) / float(len(test_y))
            self.assertAlmostEqual(pos_ratio, 1. / r)
        # ensure all instances were used in test
        self.assertEqual(len(te), n)

    def test_bootstrapped_folds_from_sizes(self):
        n = 10000
        r = 4
        n_folds = 10
        ptr = 2000
        pte = 500
        ntr = 6000
        nte = 1000
        df = pd.DataFrame({'a':np.arange(n), 'y':np.hstack([np.ones(n/r), np.zeros(n/r * (r -1))])})
        balanced_folds = BootstrapFolds(n_folds, F('y'), df, seed=1,
                                        pos_train=ptr, neg_train=ntr, pos_test=pte, neg_test=nte)
        folds = list(balanced_folds)
        self.assertEqual(len(folds), n_folds)
        te = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            te = te | set(train)
            train_y = df.y[train]
            test_y = df.y[test]
            # ensure sizes are correct
            self.assertEqual(len(train_y), ptr + ntr)
            self.assertEqual(len(test_y), pte + nte)
            self.assertEqual(sum(train_y), ptr)
            self.assertEqual(sum(test_y), pte)
        # ensure all instances were used in test (well, almost all)
        self.assertEqual(len(te), 9998) # seed is set

    def test_bootstrapped_folds_from_percents(self):
        n = 10000
        r = 4
        n_folds = 10
        train_pos_percent = .5
        test_pos_percent = .2
        train_percent = .8

        df = pd.DataFrame({'a':np.arange(n), 'y':np.hstack([np.ones(n/r), np.zeros(n/r * (r -1))])})
        balanced_folds = BootstrapFolds(n_folds, F('y'), df, seed=1,
                                        train_pos_percent=train_pos_percent,
                                        test_pos_percent=test_pos_percent,
                                        train_percent=train_percent)
        folds = list(balanced_folds)
        self.assertEqual(len(folds), n_folds)
        te = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            te = te | set(train)
            train_y = df.y[train]
            test_y = df.y[test]
            # ensure sizes are correct
            self.assertAlmostEqual(sum(train_y) / float(len(train_y)), train_pos_percent, 2)
            self.assertAlmostEqual(sum(test_y) / float(len(test_y)), test_pos_percent, 2)
        self.assertEqual(len(te), 9442) # seed is set

    def test_basic_folds(self):
        n = 10000
        n_folds = 10
        df = pd.DataFrame({'a':np.arange(n)})
        bfolds = BasicFolds(n_folds, df)
        folds = list(bfolds)
        self.assertEqual(len(folds), n_folds)
        te = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            te = te | set(test)
            train_y = df.loc[train]
            test_y = df.loc[test]
            # ensure sizes are correct
            self.assertEqual(len(train_y), n / float(n_folds) * 9)
            self.assertEqual(len(test_y), n / float(n_folds))
        # ensure all instances were used in test
        self.assertEqual(len(te), n)


if __name__ == '__main__':
    unittest.main(verbosity=2)

