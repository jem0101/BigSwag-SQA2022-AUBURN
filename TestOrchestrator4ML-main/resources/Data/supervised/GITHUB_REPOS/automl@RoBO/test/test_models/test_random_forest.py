import unittest
import numpy as np

from robo.models.random_forest import RandomForest


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = RandomForest()
        self.model.train(self.X, self.y)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)
        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)


if __name__ == "__main__":
    unittest.main()
