# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np
import scipy.sparse as sp
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, Pipeline, _name_estimators
from splearn.rdd import ArrayRDD, DictRDD


class SparkPipeline(Pipeline):

    """Distributed implementation of sklearn's pipeline node.

    Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.

    Parameters
    ----------
    steps: list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from splearn.rdd import DictRDD
    >>> from splearn.linear_model.base import SparkLinearRegression
    >>> from splearn.pipeline import SparkPipeline

    >>> class Noiser(BaseEstimator):
    >>>     def __init__(self, random_seed=42):
    >>>         np.random.seed(random_seed)
    >>>     def fit(self, Z):
    >>>         return self
    >>>     def transform(self, Z):
    >>>         f = lambda X: X + np.random.rand(*X.shape)
    >>>         if isinstance(Z, DictRDD):
    >>>             return Z.transform(f, column='X')
    >>>         else:
    >>>             return Z.transform(f)

    >>> X = np.arange(100)[:, np.newaxis]
    >>> y = np.arange(100)
    >>> X_rdd = sc.parallelize(X)
    >>> y_rdd = sc.parallelize(y)
    >>> rdd = X_rdd.zip(y_rdd)
    >>> Z = DictRDD(rdd, ('X', 'y'), 25)

    >>> pipe = SparkPipeline([('noise', Noiser()),
    >>>                       ('reg', SparkLinearRegression())])
    >>> pipe.fit(Z)
    SparkPipeline(steps=[
        ('noise', Noiser(random_seed=None)),
        ('reg', SparkLinearRegression(copy_X=True,
                                      fit_intercept=True,
                                      n_jobs=1,
                                      normalize=False)
        )])

    >>> pipe.predict(Z[:, 'X']).collect()
    [array([ 1.51878876, 2.50336579, 3.20260105, 4.41610508, 5.52531787]),
     array([ 5.56329829, 6.54787532, 7.24711057, 8.46061461, 9.5698274 ]),
     array([ 9.60780781, 10.59238484, 11.2916201, 12.50512413, 13.61433693]),
     array([ 13.65231734, 14.63689437, 15.33612963, 16.54963366, 17.65884645])]
    """

    def _pre_transform(self, Z, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Zp = Z.persist()
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Zt = transform.fit_transform(Zp, **fit_params_steps[name])
            else:
                Zt = transform.fit(Zp, **fit_params_steps[name]) \
                              .transform(Zp)
            Zp.unpersist()
            Zp = Zt.persist()
        return Zp, fit_params_steps[self.steps[-1][0]]

    def fit(self, Z, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        Z : ArrayRDD, TupleRDD or DictRDD
            Input data in blocked distributed format.

        Returns
        -------
        self : SparkPipeline
        """
        Zt, fit_params = self._pre_transform(Z, **fit_params)
        self.steps[-1][-1].fit(Zt, **fit_params)
        Zt.unpersist()
        return self

    def fit_transform(self, Z, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator."""
        Zt, fit_params = self._pre_transform(Z, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Zt, **fit_params)
        else:
            return self.steps[-1][-1].fit(Zt, **fit_params).transform(Zt)

    def score(self, Z):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score."""
        Zt = Z
        for name, transform in self.steps[:-1]:
            Zt = transform.transform(Zt)
        return self.steps[-1][-1].score(Zt)

    def get_params(self, deep=True):
        if not deep:
            return super(SparkPipeline, self).get_params(deep=False)
        else:
            out = self.named_steps.copy()
            for name, step in six.iteritems(self.named_steps):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(super(SparkPipeline, self).get_params(deep=False))
            return out

    def to_scikit(self):
        scikit_steps = []
        for name, step in self.steps:
            if hasattr(step, 'to_scikit'):
                scikit_steps.append((name, step.to_scikit()))
            else:
                scikit_steps.append((name, step))
        return Pipeline(scikit_steps)

################################################################################


def _fit_one_transformer(transformer, Z, **fit_params):
    return transformer.fit(Z, **fit_params)


def _transform_one(transformer, name, Z, transformer_weights):
    if transformer_weights is not None and name in transformer_weights:
        # if we have a weight for this transformer, muliply output
        if isinstance(Z, DictRDD):
            return transformer.transform(Z).transform(
                lambda x: x * transformer_weights[name], 'X')
        else:
            return transformer.transform(Z).map(
                lambda x: x * transformer_weights[name])
    return transformer.transform(Z)


def _fit_transform_one(transformer, name, Z, transformer_weights,
                       **fit_params):
    if transformer_weights is not None and name in transformer_weights:
        # if we have a weight for this transformer, muliply output
        if hasattr(transformer, 'fit_transform'):
            Z_transformed = transformer.fit_transform(Z, **fit_params)
        else:
            Z_transformed = transformer.fit(Z, **fit_params).transform(Z)
        # multiplication by weight
        if isinstance(Z, DictRDD):
            Z_transformed.transform(
                lambda x: x * transformer_weights[name], 'X')
        else:
            Z_transformed.map(lambda x: x * transformer_weights[name])

        return Z_transformed, transformer
    if hasattr(transformer, 'fit_transform'):
        Z_transformed = transformer.fit_transform(Z, **fit_params)
        return Z_transformed, transformer
    else:
        Z_transformed = transformer.fit(Z, **fit_params).transform(Z)
        return Z_transformed, transformer


def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


class SparkFeatureUnion(FeatureUnion):

    """TODO: rewrite docstring
    Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.
    Parameters
    ----------
    transformer_list: list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.
    n_jobs: int, optional
        Number of jobs to run in parallel (default 1).
    transformer_weights: dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
    """

    def fit(self, Z, **fit_params):
        """TODO: rewrite docstring
        Fit all transformers using X.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data, used to fit transformers.
        """
        fit_params_steps = dict((step, {})
                                for step, _ in self.transformer_list)

        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        transformers = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_fit_one_transformer)(trans, Z, **fit_params_steps[name])
            for name, trans in self.transformer_list)
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, Z, **fit_params):
        """TODO: rewrite docstring
        Fit all transformers using X, transform the data and concatenate
        results.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        return self.fit(Z, **fit_params).transform(Z)

    def transform(self, Z):
        """TODO: rewrite docstring
        Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        if isinstance(Z, DictRDD):
            X = Z[:, 'X']
        else:
            X = Z

        Zs = [_transform_one(trans, name, X, self.transformer_weights)
              for name, trans in self.transformer_list]
        X_rdd = reduce(lambda x, y: x.zip(y._rdd), Zs)
        X_rdd = X_rdd.map(flatten)
        mapper = np.hstack
        for item in X_rdd.first():
            if sp.issparse(item):
                mapper = sp.hstack
        X_rdd = X_rdd.map(lambda x: mapper(x))

        if isinstance(Z, DictRDD):
            return DictRDD([X_rdd, Z[:, 'y']],
                           columns=Z.columns,
                           dtype=Z.dtype,
                           bsize=Z.bsize)
        else:
            return X_rdd

    def get_params(self, deep=True):
        if not deep:
            return super(SparkFeatureUnion, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in six.iteritems(trans.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            out.update(super(SparkFeatureUnion, self).get_params(deep=False))
            return out

    def to_scikit(self):
        sk_transformer_list = []
        for name, transformer in self.transformer_list:
            if hasattr(transformer, 'to_scikit'):
                sk_transformer_list.append((name, transformer.to_scikit()))
            else:
                sk_transformer_list.append((name, transformer))
        return FeatureUnion(sk_transformer_list)


def make_sparkunion(*transformers):
    """Construct a FeatureUnion from the given transformers.
    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.
    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> make_union(PCA(), TruncatedSVD())    # doctest: +NORMALIZE_WHITESPACE
    FeatureUnion(n_jobs=1,
                 transformer_list=[('pca', PCA(copy=True, n_components=None,
                                               whiten=False)),
                                   ('truncatedsvd',
                                    TruncatedSVD(algorithm='randomized',
                                                 n_components=2, n_iter=5,
                                                 random_state=None, tol=0.0))],
                 transformer_weights=None)
    Returns
    -------
    f : FeatureUnion
    """
    return SparkFeatureUnion(_name_estimators(transformers))
