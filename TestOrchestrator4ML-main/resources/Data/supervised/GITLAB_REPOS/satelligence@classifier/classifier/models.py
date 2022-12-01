"""File containing all model classes necessary to run the classifier"""
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble.forest import ForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# initiate logger
MODELS_LOGGER = logging.getLogger(__name__)


class BaseMixin:
    """A mixin class for all models. This allows for each algorithm to use the
    optimise and plotting functions"""

    def plot_time_vs_accuracy(self, out_dir):
        """Plotting the training time vs the accuracy.

        Args:
            time (array): The time values to plot
            accuracy(array) The accuracy values to plot
            out_dir: THe output directory (path)

        Returns:
            Nothing
        """
        time = self.cv_df['mean_fit_time']
        accuracy = self.cv_df['mean_test_score']
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.scatter(time, accuracy)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Accuracy (-) ")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Optimisation_time.png'), dpi=300)

    def random_optimise(self, trainx, trainy, out_dir,
                        optimise_iters=10):
        """Optimization of a parameter space using random samples from the
        parameter space

        Args:
            trainx (array) :   Training dataset features (X-values)
            trainy (array) :   Training dataset outputs (class names, y values)
            out_dir (string):  The output directory path
            optimise_iters (int)  The number of iterations for the optimisation.

        Returns:
            The best performing parameter combination  (dict)

        """
        MODELS_LOGGER.info("\n####-----Optimisation----#####\n")
        MODELS_LOGGER.info("Starting Optimisation. This might take a while....")
        clf = RandomizedSearchCV(self,
                                 self.parameter_matrix,
                                 verbose=0,
                                 refit=True,
                                 cv=3,
                                 iid=True,
                                 return_train_score=True,
                                 n_jobs=-1,
                                 n_iter=optimise_iters
                                )
        clf.fit(trainx, trainy)
        MODELS_LOGGER.info("Best estimator: \n %s", clf.best_estimator_)
        self.cv_df = pd.DataFrame(clf.cv_results_)
        MODELS_LOGGER.debug("--Optimisation Results--\n %s",
                            self.cv_df[['mean_train_score', 'mean_test_score',
                                        'mean_fit_time']])
        parameter_set = self.cv_df['params'][self.cv_df['rank_test_score'] ==
                                             1].values[0]
        MODELS_LOGGER.debug("\nThe best parameter combination is:\n %s",
                            parameter_set)
        self.plot_time_vs_accuracy(out_dir)
        return parameter_set

# pylint: disable=too-many-ancestors
class RandomForest(ForestClassifier, BaseMixin):
    """RandomForest class; child of RandomForestClassifier from sk-learn"""

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForest, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=["criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"],
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        self.parameter_matrix = {'max_features': ['auto', 'sqrt', 'log2'],
                                 'max_depth': [None, 1, 3, 10, 20000]}


class XGBoost(XGBClassifier, BaseMixin):
    """XGBoost class, child of XGBoostclassifier model from sk-learn"""

    def __init__(self,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 silent=True,
                 objective="binary:logistic",
                 booster='gbtree',
                 n_jobs=1,
                 nthread=-1,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 random_state=0,
                 seed=0,
                 missing=None, **kwargs):
        super(XGBoost, self).__init__(max_depth,
                                      learning_rate,
                                      n_estimators,
                                      silent,
                                      objective,
                                      booster,
                                      n_jobs,
                                      nthread,
                                      gamma,
                                      min_child_weight,
                                      max_delta_step,
                                      subsample,
                                      colsample_bytree,
                                      colsample_bylevel,
                                      reg_alpha,
                                      reg_lambda,
                                      scale_pos_weight,
                                      base_score,
                                      random_state,
                                      seed,
                                      missing,
                                      **kwargs)
        self.parameter_matrix = {'learning_rate': [0.1, 0.2, 0.3],
                                 "n_estimators": [10, 20, 50, 100, 200],
                                 'max_depth': [1, 3, 5, 10]
                                }


class SingleClass(IsolationForest, BaseMixin):

    """Singleclass class, child of IsolationForest model from sk-learn"""


    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0
                ):
        super(SingleClass, self).__init__(n_estimators,
                                          max_samples,
                                          contamination,
                                          max_features,
                                          bootstrap,
                                          n_jobs,
                                          random_state,
                                          verbose
                                         )

        self.parameter_matrix = {"n_estimators": [10, 20, 50, 100, 200],
                                 'max_samples': ['auto', 10, 50, 100, 200],
                                 "max_features": [0.1, 0.3, 0.5, 1.0]
                                }
