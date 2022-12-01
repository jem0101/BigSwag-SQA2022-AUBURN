# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team

from __future__ import print_function


def parallel_loop(func, n_jobs=1, verbose=1):
    """run loops in parallel, if joblib is available.

    Parameters
    ----------
    func : function
        function to be executed in parallel
    n_jobs : int | None
        Number of jobs. If set to None, do not attempt to use joblib.
    verbose : int
        verbosity level

    Notes
    -----
    Execution of the main script must be guarded with `if __name__ == '__main__':` when using parallelization.
    """
    if n_jobs:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            try:
                from sklearn.externals.joblib import Parallel, delayed
            except ImportError:
                n_jobs = None

    if not n_jobs:
        if verbose:
            print('running ', func, ' serially')
        par = lambda x: list(x)
    else:
        if verbose:
            print('running ', func, ' in parallel')
        func = delayed(func)
        par = Parallel(n_jobs=n_jobs, verbose=verbose)

    return par, func
