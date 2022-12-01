﻿"""
.. index:: Neural Network Learner

***************************************
Neural Network Learner  (``neural``)
***************************************


.. index:: Neural Network Learner
.. autoclass:: Orange.classification.neural.NeuralNetworkLearner
    :members:
    :show-inheritance:

.. index:: Neural Network Classifier
.. autoclass:: Orange.classification.neural.NeuralNetworkClassifier
    :members:
    :show-inheritance:

"""


import Orange
import random
import numpy as np
import scipy.sparse
from scipy.optimize import fmin_l_bfgs_b

class _NeuralNetwork:
    def __init__(self, layers,lambda_=1, callback=None, **fmin_args):
        self.layers = layers
        self.lambda_ = lambda_
        self.callback = callback
        self.fmin_args = fmin_args

    def unfold_params(self, params):
        i0, i1, i2 = self.layers

        i = (i0 + 1) * i1

        Theta1 = params[:i].reshape((i1, i0 + 1))
        Theta2 = params[i:].reshape((i2, i1 + 1))

        return Theta1, Theta2

    def cost_grad(self, params):
        Theta1, Theta2 = self.unfold_params(params)

        if self.callback:
            self.Theta1 = Theta1
            self.Theta2 = Theta2
            self.callback(self)

        # Feedforward Propagation
        m, n = self.X.shape

        a1 = self.X
        z2 = a1.dot(Theta1.T)
        a2 = np.column_stack((np.ones(m), _sigmoid(z2)))
        z3 = a2.dot(Theta2.T)
        a3 = _sigmoid(z3)

        # Cost
        J = np.sum(-self.y * np.log(a3) - (1 - self.y) * np.log(1 - a3)) / m

        t1 = Theta1.copy()
        t1[:, 0] = 0
        t2 = Theta2.copy()
        t2[:, 0] = 0

        # regularization
        reg = np.dot(t1.flat, t1.flat)
        reg += np.dot(t2.flat, t2.flat)
        J += float(self.lambda_) * reg / (2.0 * m)

        # Grad
        d3 = a3 - self.y
        d2 = d3.dot(Theta2)[:, 1:] * _sigmoid_gradient(z2)

        D2 = a2.T.dot(d3).T / m
        D1 = a1.T.dot(d2).T / m

        # regularization
        D2 += t2 * (float(self.lambda_) / m)
        D1 += t1 * (float(self.lambda_) / m)

        return J, np.hstack((D1.flat, D2.flat))

    def fit(self, X, y):
        i0, i1, i2 = self.layers

        m, n = X.shape
        n_params = i1 * (i0 + 1) + i2 * (i1 + 1)
        eps = np.sqrt(6) / np.sqrt(i0 + i2)
        initial_params = np.random.randn(n_params) * 2 * eps - eps

        self.X = self.append_ones(X)
        self.y = y

        params, _, _ = fmin_l_bfgs_b(self.cost_grad, initial_params, **self.fmin_args)

        self.Theta1, self.Theta2 = self.unfold_params(params)

    def predict(self, X):
        m, n = X.shape
        
        a2 = _sigmoid(self.append_ones(X).dot(self.Theta1.T))
        a3 = _sigmoid(np.column_stack((np.ones(m), a2)).dot(self.Theta2.T))

        return a3

    def append_ones(self, X):
        m, n = X.shape
        if scipy.sparse.issparse(X):
            return scipy.sparse.hstack((np.ones((m, 1)), X)).tocsr()
        else:
            return np.column_stack((np.ones(m), X))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _sigmoid_gradient(x):
    sx = _sigmoid(x)
    return sx * (1 - sx)


class NeuralNetworkLearner(Orange.classification.Learner):
    """
    NeuralNetworkLearner implements a multilayer perceptron. Learning is performed by minimizing an L2-regularized
    cost function with scipy's implementation of L-BFGS. The current implementations is limited to a single
    hidden layer. 

    Regression is currently not supported.

    :param name: learner name.
    :type name: string

    :param n_mid: Number of nodes in the hidden layer
    :type n_mid: int

    :param reg_fact: Regularization factor.
    :type reg_fact: float

    :param max_iter: Maximum number of iterations.
    :type max_iter: int

    :param normalize: Normalize the data prior to learning (subtract each column by the mean and divide by the standard deviation)
    :type normalize: bool

    :rtype: :class:`~NeuralNetworkLearner` or
            :class:`~NeuralNetworkClassifier`
    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)

        if data is None:   
            return self
        else:
            self.__init__(**kwargs)
            return self(data,weight)

    def __init__(self, name="NeuralNetwork", n_mid=10, reg_fact=1,
                 max_iter=300, normalize=True, rand=None):
        self.name = name
        self.n_mid = n_mid
        self.reg_fact = reg_fact
        self.max_iter = max_iter
        self.rand = rand
        self.normalize = normalize

        if not self.rand:
            self.rand = random.Random(42)
        np.random.seed(self.rand.randint(0,10000))

    def __call__(self,data,weight=0):
        """
        Learn from the given table of data instances.
        
        :param instances: data for learning.
        :type instances: :class:`Orange.data.Table`

        :param weight: weight.
        :type weight: int

        :rtype: :class:`~NeuralNetworkClassifier`
        """

        #converts attribute data
        X = data.to_numpy()[0]

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        if self.normalize:
            X -= mean
            X[:, std > 0] /= std[std > 0]

        #converts multi-target or single-target classes to numpy
        if any(isinstance(var, Orange.feature.Continuous)
               for var in (data.domain.class_vars or [data.domain.class_var])):
            raise ValueError("non-discrete classes not supported")

        if data.domain.class_vars:
            cvals = [len(cv.values) if len(cv.values) > 2 else 1 for cv in data.domain.class_vars]
            Y = np.zeros((len(data), sum(cvals)))
            cvals = [0]+[sum(cvals[0:i+1]) for i in xrange(len(cvals))]  

            for i in xrange(len(data)):
                for j in xrange(len(cvals)-1):
                    if cvals[j+1] - cvals[j] > 2:
                        Y[i, cvals[j] + int(data[i].get_classes()[j])] = 1.0
                    else:
                        Y[i, cvals[j]] = float(data[i].get_classes()[j])
        else:
            y = np.array([int(d.get_class()) for d in data])
            n_classes = len(data.domain.class_var.values)
            if n_classes > 2:
                Y = np.eye(n_classes)[y]
            else:
                Y = y[:,np.newaxis]
       
        #initializes neural networks
        nn =  _NeuralNetwork([len(X[0]), self.n_mid,len(Y[0])], lambda_=self.reg_fact, maxfun=self.max_iter, iprint=-1)

        with np.errstate(over="ignore"):
            nn.fit(X, Y)

        return NeuralNetworkClassifier(domain=data.domain, nn=nn, normalize=self.normalize, mean=mean, std=std)


class NeuralNetworkClassifier(Orange.classification.Classifier):
    """
    Classifier induced by the :obj:`NeuralNetworkLearner`.
    """

    def __init__(self, domain, nn, normalize, mean, std, **kwargs):
        self.domain = domain
        if domain.class_vars:
            self.class_vars = domain.class_vars
            self.class_var = None
        else:
            self.class_var = domain.class_var
        self.nn = nn
        self.normalize = normalize
        self.mean = mean
        self.std = std

        for name, val in kwargs.items():
            setattr(self, name, val)

    def __reduce__(self):
        return (type(self),
                (self.domain, self.nn, self.normalize, self.mean, self.std),
                dict(self.__dict__))

    def __call__(self, example, result_type=Orange.core.GetValue):
        """
        :param example: instance to be classified.
        :type example: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        example = Orange.data.Instance(self.domain, example)
        # transform example to numpy
        if not self.domain.class_vars:
            example = [example[i] for i in range(len(example)-1)]
        input = np.array([[float(e) for e in example]])

        if self.normalize:
            input -= self.mean
            input[:, self.std > 0] /= self.std[self.std > 0]

        with np.errstate(over="ignore"):
            results = self.nn.predict(input)

        # transform results from numpy
        results = results.tolist()[0]
        if len(results) == 1:
            prob_positive = results[0]
            results = [1 - prob_positive, prob_positive]
        mt_prob = []
        mt_value = []
          
        if self.domain.class_vars:
            cvals = [len(cv.values) if len(cv.values) > 2 else 1 for cv in self.domain.class_vars]
            cvals = [0] + [sum(cvals[0:i]) for i in xrange(1, len(cvals) + 1)]

            for cls in xrange(len(self.domain.class_vars)):
                if cvals[cls+1]-cvals[cls] > 2:
                    cprob = Orange.statistics.distribution.Discrete(results[cvals[cls]:cvals[cls+1]])
                    cprob.normalize()
                else:
                    r = results[cvals[cls]]
                    cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])

                mt_prob.append(cprob)
                mt_value.append(Orange.data.Value(self.domain.class_vars[cls], cprob.values().index(max(cprob))))

        else:
            assert len(self.domain.class_var.values) == len(results)
            cprob = Orange.statistics.distribution.Discrete(results)
            cprob.variable = self.domain.class_var
            cprob.normalize()
            mt_prob = cprob
            mt_value = Orange.data.Value(self.domain.class_var, cprob.values().index(max(cprob)))

        if result_type == Orange.core.GetValue: return tuple(mt_value) if self.domain.class_vars else mt_value
        elif result_type == Orange.core.GetProbabilities: return tuple(mt_prob) if self.domain.class_vars else mt_prob
        else: 
            return [tuple(mt_value), tuple(mt_prob)] if self.domain.class_vars else [mt_value, mt_prob] 

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('wine')
    l1 = NeuralNetworkLearner(n_mid=40, reg_fact=1, max_iter=200)

#    c1 = l1(data)
#    print c1(data[0], 3), data[0]

    l2 = Orange.classification.bayes.NaiveLearner()
    res = Orange.evaluation.testing.cross_validation([l1, l2],data, 5)
    scores = Orange.evaluation.scoring.CA(res)
    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
