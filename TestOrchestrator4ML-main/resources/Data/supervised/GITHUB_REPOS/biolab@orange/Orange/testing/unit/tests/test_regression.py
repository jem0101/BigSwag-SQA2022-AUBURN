import sys
import os.path
try:
    import unittest2 as unittest
except:
    import unittest
import subprocess

import Orange
from Orange.utils import environ
from Orange.testing import testing


class TestRegression(unittest.TestCase):
    PLATFORM = sys.platform
    PYVERSION = sys.version[:3]
    STATES = ["OK", "timedout", "changed", "random", "error", "crash"]
    maxDiff = None

    def __init__(self, methodName='runTest'):
        super(TestRegression, self).__init__(methodName)
        self.orange_dir = os.path.dirname(Orange.__file__)
        self.orange_dir = os.path.join(self.orange_dir, '..')
        self.orange_dir = os.path.realpath(self.orange_dir)

    def setUp(self):
        sys.path.append(self.orange_dir)

    def tearDown(self):
        del sys.path[-1]

    def test_regression_on(self, roottest, indir, outdir, name):
        for state in TestRegression.STATES:
            remname = "%s/%s.%s.%s.%s.txt" % \
                                (outdir, name, TestRegression.PLATFORM, \
                                 TestRegression.PYVERSION, state)
            if os.path.exists(remname):
                os.remove(remname)

        # Add the current dir to PYTHONPATH because the cwd will
        # be changed in subprocess call.
        cwd = os.getcwd()
        env = dict(os.environ)
        pypath = env.get("PYTHONPATH", "")
        if pypath:
            pypath = os.path.pathsep.join([cwd, pypath])
        else:
            pypath = cwd
        env["PYTHONPATH"] = pypath

        p = subprocess.Popen([sys.executable,
                              os.path.join(roottest, "xtest_one.py"),
                              name, "1", outdir],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=env,
                              cwd=indir)

        stdout, stderr = p.communicate()
        rv = stdout.strip().lower()
        if rv == 'error':
            self.assertEqual(stderr.split('\n'), [])
        elif rv == 'changed':
            expected_results = self.get_expected_results(outdir, name)
            actual_results = self.get_actual_results(outdir, name)
            self.assertEqual(actual_results, expected_results)

        self.assertEqual(rv, "ok", "Regression test %s: %s" % (rv, name) \
                        if stderr == "" else \
                        "Regression test %s: %s\n\n%s" % (rv, name, stderr))

        self.assertEqual(p.wait(), 0,
                         "Test script exited with a non zero error code.")

    def get_expected_results(self, outputdir, name):
        expected_results = "%s/%s.%s.%s.txt" % (outputdir, name, sys.platform, sys.version[:3])
        if not os.path.exists(expected_results):
            expected_results = "%s/%s.%s.txt" % (outputdir, name, sys.platform)
            if not os.path.exists(expected_results):
                expected_results = "%s/%s.txt" % (outputdir, name)

        with open(expected_results, 'r') as results:
            return results.read().split('\n')

    def get_actual_results(self, outputdir, name):
        for state in TestRegression.STATES:
            actual_results = "%s/%s.%s.%s.%s.txt" % (
                outputdir, name, TestRegression.PLATFORM,
                TestRegression.PYVERSION, state)

            if os.path.exists(actual_results):
                with open(actual_results, 'r') as results:
                    return results.read().split('\n')


root = os.path.normpath(os.path.join(environ.install_dir, ".."))
roottest = os.path.join(root, "Orange/testing/regression")

dirs = [("tests", "Orange/testing/regression/tests"),
        ("tests_20", "Orange/testing/regression/tests_20"),
        ("tutorial", "docs/tutorial/rst/code"),
        ("reference", "docs/reference/rst/code")]

for dirname, indir in dirs:
    indir = os.path.join(root, indir)
    outdir = "%s/results_%s" % (roottest, dirname)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    exclude = os.path.join(indir, "exclude-from-regression.txt")
    dont_test = [x.strip() for x in file(exclude).readlines()] if \
                                        os.path.exists(exclude) else []
    test_set = []
    names = sorted([name for name in os.listdir(indir) if \
                    name[-3:] == ".py"and name not in dont_test])

    for name in names:
        if not os.path.exists(os.path.join(outdir, name + ".txt")):
            # past result not available
            test_set.append((name, "new"))
        else:
            # past result available
            for state in TestRegression.STATES:
                if os.path.exists("%s/%s.%s.%s.%s.txt" % \
                               (outdir, name, TestRegression.PLATFORM, \
                                TestRegression.PYVERSION, state)):
                    test_set.append((name, state))
                    # current result already on disk
                    break
            else:
                if os.path.exists("%s/%s.%s.%s.random1.txt" % \
                                  (outdir, name, TestRegression.PLATFORM, \
                                   TestRegression.PYVERSION)):
                    test_set.append((name, "random"))
                else:
                    test_set.append((name, "OK"))

    for name, last_res in test_set:
        newname, func = testing._expanded(TestRegression.test_regression_on,
                                           "%s_%s" % (dirname, name[:-3]),
                                           (roottest, indir, outdir, name))
        setattr(TestRegression, newname, func)

setattr(TestRegression, "test_regression_on", None)
func = None

if __name__ == "__main__":
    unittest.main()
