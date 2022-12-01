import unittest

import george
import numpy as np
import scipy.linalg as spla

from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)

        self.kernel = george.kernels.Matern52Kernel(np.ones(self.X.shape[1]),
                                                    ndim=self.X.shape[1])

        prior = TophatPrior(-2, 2)
        self.model = GaussianProcess(self.kernel, prior=prior,
                                     normalize_input=False,
                                     normalize_output=False)
        self.model.train(self.X, self.y, do_optimize=False)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

        m, v = self.model.predict(X_test, full_cov=True)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 2
        assert v.shape[0] == X_test.shape[0]
        assert v.shape[1] == X_test.shape[0]

        K_zz = self.kernel.get_value(X_test)
        K_zx = self.kernel.get_value(X_test, self.X)
        K_nz = self.kernel.get_value(self.X) + self.model.noise * np.eye(self.X.shape[0])
        inv = spla.inv(K_nz)
        K_zz_x = K_zz - np.dot(K_zx, np.inner(inv, K_zx))
        assert np.mean((K_zz_x - v) ** 2) < 10e-5

    def test_sample_function(self):
        X_test = np.random.rand(8, 2)
        n_funcs = 3
        funcs = self.model.sample_functions(X_test, n_funcs=n_funcs)

        assert len(funcs.shape) == 2
        assert funcs.shape[0] == n_funcs
        assert funcs.shape[1] == X_test.shape[0]

    def test_predict_variance(self):
        x_test1 = np.random.rand(1, 2)
        x_test2 = np.random.rand(10, 2)
        var = self.model.predict_variance(x_test1, x_test2)
        assert len(var.shape) == 2
        assert var.shape[0] == x_test2.shape[0]
        assert var.shape[1] == x_test1.shape[0]

    def test_nll(self):
        theta = np.array([0.2, 0.2, 0.001])
        nll = self.model.nll(theta)

    def test_optimize(self):
        theta = self.model.optimize()
        # Hyperparameters are 2 length scales + noise
        assert theta.shape[0] == 3

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)
        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)
        assert inc_val == self.y[b]


if __name__ == "__main__":
    unittest.main()
