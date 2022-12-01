# -*- coding: utf-8 -*-
"""
SVM (PSO, global timeout)
================================
"""
# Future
from __future__ import absolute_import, division, print_function, \
    unicode_literals, with_statement

# Third Party
from sklearn import cross_validation, datasets, svm

# First Party
from metaopt.core.paramspec.util import param
from metaopt.core.returnspec.util.decorator import maximize

@maximize("Score")
@param.float("C-Exp", interval=[0, 4])
@param.float("Gamma-Exp", interval=[-6, 0])
def f(C_exp, gamma_exp):
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)

    C = 10 ** C_exp
    gamma = 10 ** gamma_exp
    clf = svm.SVC(C=C, gamma=gamma)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


def main():
    from metaopt.core.optimize.optimize import optimize
    from metaopt.optimizer.pso import PSOOptimizer

    from metaopt.plugin.print.status import StatusPrintPlugin
    from metaopt.plugin.visualization.landscape import VisualizeLandscapePlugin
    from metaopt.plugin.visualization.best_fitness import VisualizeBestFitnessPlugin

    timeout = 10
    optimizer = PSOOptimizer()

    visualize_landscape_plugin = VisualizeLandscapePlugin()
    visualize_best_fitness_plugin = VisualizeBestFitnessPlugin()

    plugins = [
        StatusPrintPlugin(),
        visualize_landscape_plugin,
        visualize_best_fitness_plugin
    ]

    optimum = optimize(f=f, timeout=timeout, optimizer=optimizer,
                       plugins=plugins)

    print("The optimal parameters are %s." % str(optimum))

    visualize_landscape_plugin.show_surface_plot()
    visualize_landscape_plugin.show_image_plot()

    visualize_best_fitness_plugin.show_fitness_invocations_plot()
    visualize_best_fitness_plugin.show_fitness_time_plot()


if __name__ == '__main__':
    main()
