# Future
from __future__ import absolute_import, division, print_function, \
    unicode_literals, with_statement

# Third Party
import nose
from mock import Mock
from nose.tools import raises

# First Party
from metaopt.core.optimize.optimize import NoParamSpecError, custom_optimize

@raises(NoParamSpecError)
def test_custom_optimize_given_no_param_spec_complains():
    invoker = Mock()

    def f(x, y):
        pass

    custom_optimize(f, invoker)

if __name__ == '__main__':
    nose.runmodule()
