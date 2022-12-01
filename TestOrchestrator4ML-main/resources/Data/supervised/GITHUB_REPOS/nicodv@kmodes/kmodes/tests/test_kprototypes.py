"""
Tests for k-prototypes clustering algorithm
"""

import pickle
import unittest

import numpy as np
from sklearn.utils.testing import assert_equal

from kmodes import kprototypes
from kmodes.tests.test_kmodes import assert_cluster_splits_equal
from kmodes.util.dissim import ng_dissim

STOCKS = np.array([
    [738.5, 'tech', 'USA'],
    [369.5, 'nrg', 'USA'],
    [368.2, 'tech', 'USA'],
    [346.7, 'tech', 'USA'],
    [343.5, 'fin', 'USA'],
    [282.4, 'fin', 'USA'],
    [282.1, 'tel', 'CN'],
    [279.7, 'cons', 'USA'],
    [257.2, 'cons', 'USA'],
    [205.2, 'tel', 'USA'],
    [192.1, 'tech', 'USA'],
    [195.7, 'nrg', 'NL']
])
STOCKS2 = np.array([
    [134.1, 'fin', 'USA'],
    [190.2, 'cons', 'USA'],
    [389.1, 'nrg', 'CA'],
    [150.4, 'mat', 'USA']
])


class TestKProtoTypes(unittest.TestCase):

    def test_pickle(self):
        obj = kprototypes.KPrototypes()
        s = pickle.dumps(obj)
        assert_equal(type(pickle.loads(s)), obj.__class__)

    def test_kprotoypes_categoricals_stocks(self):
        # Number/index of categoricals does not make sense
        kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
        with self.assertRaises(AssertionError):
            kproto.fit_predict(STOCKS, categorical=[1, 3])
        with self.assertRaises(AssertionError):
            kproto.fit_predict(STOCKS, categorical=[0, 1, 2])
        result = kproto.fit(STOCKS[:, :2], categorical=1)
        self.assertIsInstance(result, kprototypes.KPrototypes)

    def test_kprotoypes_wrong_categorical_type(self):
        kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
        with self.assertRaises(AssertionError):
            kproto.fit_predict(STOCKS, categorical={1, 2})

    def test_kprotoypes_huang_stocks(self):
        kproto_huang = kprototypes.KPrototypes(n_clusters=4, n_init=1,
                                               init='Huang', verbose=2,
                                               random_state=42)
        # Untrained model
        with self.assertRaises(AssertionError):
            kproto_huang.predict(STOCKS, categorical=[1, 2])
        result = kproto_huang.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_huang_stocks_parallel(self):
        kproto_huang = kprototypes.KPrototypes(n_clusters=4, n_init=4,
                                               init='Huang', verbose=2,
                                               random_state=42, n_jobs=4)
        # Untrained model
        with self.assertRaises(AssertionError):
            kproto_huang.predict(STOCKS, categorical=[1, 2])
        result = kproto_huang.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_cao_stocks(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao',
                                             verbose=2, random_state=42)
        result = kproto_cao.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_predict_stocks(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao',
                                             verbose=2, random_state=42)
        kproto_cao = kproto_cao.fit(STOCKS, categorical=[1, 2])
        result = kproto_cao.predict(STOCKS2, categorical=[1, 2])
        expected = np.array([1, 1, 3, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprototypes_predict_unfitted(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao',
                                             verbose=2, random_state=42)
        with self.assertRaises(AssertionError):
            kproto_cao.predict(STOCKS)
        with self.assertRaises(AttributeError):
            kproto_cao.cluster_centroids_

    def test_kprotoypes_random_stocks(self):
        kproto_random = kprototypes.KPrototypes(n_clusters=4, init='random',
                                                verbose=2)
        result = kproto_random.fit(STOCKS, categorical=[1, 2])
        self.assertIsInstance(result, kprototypes.KPrototypes)

    def test_kprotoypes_init_stocks(self):
        # Wrong order
        init_vals = [
            np.array([[3, 2],
                      [0, 2],
                      [3, 2],
                      [2, 2]]),
            np.array([[356.975],
                      [275.35],
                      [738.5],
                      [197.667]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=2, init=init_vals,
                                              verbose=2)
        with self.assertRaises(AssertionError):
            kproto_init.fit_predict(STOCKS, categorical=[1, 2])

        # Wrong number of clusters
        init_vals = [
            np.array([356.975, 275.35, 738.5, 197.667, 0.]),
            np.array([[3, 2],
                      [0, 2],
                      [3, 2],
                      [2, 2]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals,
                                              verbose=2)
        with self.assertRaises(AssertionError):
            kproto_init.fit_predict(STOCKS, categorical=[1, 2])

        # Wrong number of attributes
        init_vals = [
            np.array([356.975, 275.35, 738.5, 197.667]),
            np.array([3, 0, 3, 2])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals,
                                              verbose=2)
        with self.assertRaises(AssertionError):
            kproto_init.fit_predict(STOCKS, categorical=[1, 2])

        init_vals = [
            np.array([[356.975],
                      [275.35],
                      [738.5],
                      [197.667]]),
            np.array([[3, 2],
                      [0, 2],
                      [3, 2],
                      [2, 2]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals,
                                              verbose=2, random_state=42)
        result = kproto_init.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_missings(self):
        init_vals = [
            np.array([[356.975],
                      [275.35],
                      [738.5],
                      [np.NaN]]),
            np.array([[3, 2],
                      [0, 2],
                      [3, 2],
                      [2, 2]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals,
                                              verbose=2)
        with self.assertRaises(ValueError):
            kproto_init.fit_predict(STOCKS, categorical=[1, 2])

    def test_kprototypes_unknowninit_soybean(self):
        kproto = kprototypes.KPrototypes(n_clusters=4, init='nonsense',
                                         verbose=2)
        with self.assertRaises(NotImplementedError):
            kproto.fit(STOCKS, categorical=[1, 2])

    def test_kprotoypes_not_stuck_initialization(self):
        init_problem = np.array([
            [0, 'Regular'],
            [0, 'Regular'],
            [0, 'Regular'],
            [0, np.NaN],
            [-0.5, 'Regular'],
            [-0.5, 'Regular'],
            [0, np.NaN],
            [0, 'Regular'],
            [0, 'Regular'],
            [0, 'Slim'],
            [0, 'Regular'],
            [0, 'Regular'],
            [0.5, 'Regular'],
            [-0.5, 'Regular'],
            [0.5, 'Regular'],
            [0.5, 'Slim'],
            [0, 'Regular'],
            [0.5, 'Regular'],
            [0, 'Regular'],
            [-0.5, 'Regular'],
            [0, np.NaN],
            [0, np.NaN],
            [0, 'Regular'],
            [0, 'Regular'],
            [0, 'Regular']
        ])
        kproto_cao = kprototypes.KPrototypes(n_clusters=6, init='Cao',
                                             verbose=2, random_state=42)
        kproto_cao = kproto_cao.fit(init_problem, categorical=[1])
        self.assertTrue(hasattr(kproto_cao, 'cluster_centroids_'))

    def test_kprotoypes_n_nclusters(self):
        data = np.array([
            [0., 'Regular'],
            [0., 'Regular'],
            [0., 'Slim']
        ])
        kproto_cao = kprototypes.KPrototypes(n_clusters=6, init='Cao',
                                             verbose=2, random_state=42)
        with self.assertRaises(AssertionError):
            kproto_cao.fit_predict(data, categorical=[1])

    def test_kprotoypes_nunique_nclusters(self):
        data = np.array([
            [0., 'Regular'],
            [0., 'Regular'],
            [0., 'Regular'],
            [1., 'Slim'],
            [1., 'Slim'],
            [1., 'Slim']
        ])
        kproto_cao = kprototypes.KPrototypes(n_clusters=6, init='Cao',
                                             verbose=2, random_state=42)
        kproto_cao.fit_predict(data, categorical=[1])
        # Check if there are only 2 clusters.
        self.assertEqual(kproto_cao.cluster_centroids_.shape[1], 2)

    def test_kprotoypes_impossible_init(self):
        data = np.array([
            [0., 'Regular'],
            [0., 'Regular'],
            [0., 'Regular'],
            [0., 'Slim'],
            [0., 'Slim'],
            [0., 'Slim']
        ])
        kproto_cao = kprototypes.KPrototypes(n_clusters=2, init='Cao',
                                             verbose=2, random_state=42)
        with self.assertRaises(ValueError):
            kproto_cao.fit_predict(data, categorical=[1])

    def test_kprotoypes_no_categoricals(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=6, init='Cao',
                                             verbose=2, random_state=42)
        with self.assertRaises(NotImplementedError):
            kproto_cao.fit(STOCKS, categorical=[])

    def test_kprotoypes_huang_stocks_ng(self):
        kproto_huang = kprototypes.KPrototypes(n_clusters=4, n_init=1,
                                               init='Huang', verbose=2,
                                               cat_dissim=ng_dissim,
                                               random_state=42)
        # Untrained model
        with self.assertRaises(AssertionError):
            kproto_huang.predict(STOCKS, categorical=[1, 2])
        result = kproto_huang.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_cao_stocks_ng(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao',
                                             verbose=2, cat_dissim=ng_dissim,
                                             random_state=42)
        result = kproto_cao.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_predict_stocks_ng(self):
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao',
                                             verbose=2, cat_dissim=ng_dissim,
                                             random_state=42)
        kproto_cao = kproto_cao.fit(STOCKS, categorical=[1, 2])
        result = kproto_cao.predict(STOCKS2, categorical=[1, 2])
        expected = np.array([1, 1, 3, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprotoypes_init_stocks_ng(self):
        init_vals = [
            np.array([[356.975],
                      [275.35],
                      [738.5],
                      [197.667]]),
            np.array([[3, 2],
                      [0, 2],
                      [3, 2],
                      [2, 2]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals,
                                              verbose=2, cat_dissim=ng_dissim,
                                              random_state=42)
        result = kproto_init.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kprototypes_ninit(self):
        kmodes = kprototypes.KPrototypes(n_init=10, init='Huang')
        self.assertEqual(kmodes.n_init, 10)
        kmodes = kprototypes.KPrototypes(n_init=10, init='Cao')
        self.assertEqual(kmodes.n_init, 10)
        kmodes = kprototypes.KPrototypes(n_init=10, init=[np.array([]), np.array([])])
        self.assertEqual(kmodes.n_init, 1)
