# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2015 SCoT Development Team

"""
Summary
-------
Tools for basic data manipulation.
"""

from __future__ import division

import numpy as np

from .utils import check_random_state


def atleast_3d(x):
    x = np.asarray(x)
    if x.ndim >= 3:
        return x
    elif x.ndim == 2:
        return x[np.newaxis, ...]
    else:
        return x[np.newaxis, np.newaxis, :]


def cut_segments(x2d, tr, start, stop):
    """Cut continuous signal into segments.

    Parameters
    ----------
    x2d : array, shape (m, n)
        Input data with m signals and n samples.
    tr : list of int
        Trigger positions.
    start : int
        Window start (offset relative to trigger).
    stop : int
        Window end (offset relative to trigger).

    Returns
    -------
    x3d : array, shape (len(tr), m, stop-start)
        Segments cut from data. Individual segments are stacked along the first
        dimension.

    See also
    --------
    cat_trials : Concatenate segments.

    Examples
    --------
    >>> data = np.random.randn(5, 1000)  # 5 channels, 1000 samples
    >>> tr = [750, 500, 250]  # three segments
    >>> x3d = cut_segments(data, tr, 50, 100)  # each segment is 50 samples
    >>> x3d.shape
    (3, 5, 50)
    """
    if start != int(start):
        raise ValueError("start index must be an integer")
    if stop != int(stop):
        raise ValueError("stop index must be an integer")

    x2d = np.atleast_2d(x2d)
    tr = np.asarray(tr, dtype=int).ravel()
    win = np.arange(start, stop, dtype=int)
    return np.concatenate([x2d[np.newaxis, :, t + win] for t in tr])


def cat_trials(x3d):
    """Concatenate trials along time axis.

    Parameters
    ----------
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples.

    Returns
    -------
    x2d : array, shape (m, t * n)
        Trials are concatenated along the second axis.

    See also
    --------
    cut_segments : Cut segments from continuous data.

    Examples
    --------
    >>> x = np.random.randn(6, 4, 150)
    >>> y = cat_trials(x)
    >>> y.shape
    (4, 900)
    """
    x3d = atleast_3d(x3d)
    t = x3d.shape[0]
    return np.concatenate(np.split(x3d, t, 0), axis=2).squeeze(0)


def dot_special(x2d, x3d):
    """Segment-wise dot product.

    This function calculates the dot product of x2d with each trial of x3d.

    Parameters
    ----------
    x2d : array, shape (p, m)
        Input argument.
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples. The dot
        product with x2d is calculated for each trial.

    Returns
    -------
    out : array, shape (t, p, n)
        Dot product of x2d with each trial of x3d.

    Examples
    --------
    >>> x = np.random.randn(6, 40, 150)
    >>> a = np.ones((7, 40))
    >>> y = dot_special(a, x)
    >>> y.shape
    (6, 7, 150)
    """
    x3d = atleast_3d(x3d)
    x2d = np.atleast_2d(x2d)
    return np.concatenate([x2d.dot(x3d[i, ...])[np.newaxis, ...]
                           for i in range(x3d.shape[0])])


def randomize_phase(data, random_state=None):
    """Phase randomization.

    This function randomizes the spectral phase of the input data along the
    last dimension.

    Parameters
    ----------
    data : array
        Input array.

    Returns
    -------
    out : array
        Array of same shape as data.

    Notes
    -----
    The algorithm randomizes the phase component of the input's complex Fourier
    transform.

    Examples
    --------
    .. plot::
        :include-source:

        from pylab import *
        from scot.datatools import randomize_phase
        np.random.seed(1234)
        s = np.sin(np.linspace(0,10*np.pi,1000))
        x = np.vstack([s, np.sign(s)])
        y = randomize_phase(x)
        subplot(2,1,1)
        title('Phase randomization of sine wave and rectangular function')
        plot(x.T + [1.5, -1.5]), axis([0,1000,-3,3])
        subplot(2,1,2)
        plot(y.T + [1.5, -1.5]), axis([0,1000,-3,3])
        plt.show()
    """
    rng = check_random_state(random_state)
    data = np.asarray(data)
    data_freq = np.fft.rfft(data)
    data_freq = np.abs(data_freq) * np.exp(1j*rng.random_sample(data_freq.shape)*2*np.pi)
    return np.fft.irfft(data_freq, data.shape[-1])


def acm(x, l):
    """Compute autocovariance matrix at lag l.

    This function calculates the autocovariance matrix of `x` at lag `l`.

    Parameters
    ----------
    x : array, shape (n_trials, n_channels, n_samples)
        Signal data (2D or 3D for multiple trials)
    l : int
        Lag

    Returns
    -------
    c : ndarray, shape = [nchannels, n_channels]
        Autocovariance matrix of `x` at lag `l`.
    """
    x = atleast_3d(x)

    if l > x.shape[2]-1:
        raise AttributeError("lag exceeds data length")

    ## subtract mean from each trial
    #for t in range(x.shape[2]):
    #    x[:, :, t] -= np.mean(x[:, :, t], axis=0)

    if l == 0:
        a, b = x, x
    else:
        a = x[:, :, l:]
        b = x[:, :, 0:-l]

    c = np.zeros((x.shape[1], x.shape[1]))
    for t in range(x.shape[0]):
        c += a[t, :, :].dot(b[t, :, :].T) / a.shape[2]
    c /= x.shape[0]

    return c.T
