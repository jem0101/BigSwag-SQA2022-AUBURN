"""Linear systems"""

import numpy as np

__all__ = ["tf2zp",  'eqtflength', 'latc2tf', 'latcfilt',
    'ss2zpk', 'tf2sos', 'tf2ss', 'tf2zpk', 'zpk2ss', 'zpk2tf']


"""to be done

latc2tf    Convert lattice filter parameters to transfer function form
polyscale    Scale roots of polynomial
polystab    Stabilize polynomial
residuez    z-transform partial-fraction expansion
sos2ss    Convert digital filter second-order section parameters to state-space form
sos2tf    Convert digital filter second-order section data to transfer function form
sos2zp    Convert digital filter second-order section parameters to zero-pole-gain form
ss2sos    Convert digital filter state-space parameters to second-order sections form
ss2tf    Convert state-space filter parameters to transfer function form
ss2zp    Convert state-space filter parameters to zero-pole-gain form
tf2sos    Convert digital filter transfer function data to second-order sections form
tf2ss    Convert transfer function filter parameters to state-space form
zp2sos    Convert zero-pole-gain filter parameters to second-order sections form
zp2ss    Convert zero-pole-gain filter parameters to state-space form
zp2tf    Convert zero-pole-gain filter parameters to transfer function form
"""


def tf2zp(b,a):
    """Convert transfer function filter parameters to zero-pole-gain form

    Find the zeros, poles, and gains of this continuous-time system:

    .. warning:: b and a must have the same length.

    ::

    
        from spectrum import tf2zp
        b = [2,3,0]
        a = [1, 0.4, 1]
        [z,p,k] = tf2zp(b,a)          % Obtain zero-pole-gain form
        z =
            1.5
            0
        p =
           -0.2000 + 0.9798i
            -0.2000 - 0.9798i
        k =
           2

    :param b: numerator
    :param a: denominator
    :param fill: If True, check that the length of a and b are the same. If not, create a copy of the shortest element and append zeros to it.
    :return: z (zeros), p (poles), g (gain)


    Convert transfer function f(x)=sum(b*x^n)/sum(a*x^n) to
    zero-pole-gain form f(x)=g*prod(1-z*x)/prod(1-p*x)

    .. todo:: See if tf2ss followed by ss2zp gives better results.  These
        are available from the control system toolbox.  Note that
        the control systems toolbox doesn't bother, but instead uses

    .. seealso:: scipy.signal.tf2zpk, which gives the same results but uses a different
        algorithm (z^-1 instead of z).
    """
    from numpy import roots
    assert len(b) == len(a), "length of the vectors a and b must be identical. fill with zeros if needed."

    g = b[0] / a[0]
    z = roots(b)
    p = roots(a)

    return z, p, g


def zp2tf(z,p,k):
    print("Use zpk2tf instead of zp2tf function")
    return zpk2tf(z,p,k)


def eqtflength(b,a):
    """Given two list or arrays, pad with zeros the shortest array

    :param b: list or array
    :param a: list or array


    .. doctest::

        >>> from spectrum.transfer import eqtflength
        >>> a = [1,2]
        >>> b = [1,2,3,4]
        >>> a, b, = eqtflength(a,b)

    """
    d = abs(len(b)-len(a))
    if d != 0:
        if len(a) > len(b):
            try:
                b.extend([0.]*d)
            except:
                b = np.append(b, [0]*d)
        elif len(b)>len(a):
            try:
                a.extend([0.]*d)
            except:
                a = np.append(a, [0]*d)
        return b,a
    else:
        return b,a

'''
def tf2latc(num=[1.], den=[1.]):
    """Convert transfer function filter parameters to lattice filter form"""

    if len(num) == 1:
        k, v = allpole2latc(num, den)

def allpole2latc(num, den):
    from spectrum.linear_prediction import poly2rc
    # All-pole filter, simply call poly2rc
    k = poly2rc(den)
    #v = [num;numpy.zeros(size(v))];
    #return k, v
'''


def latc2tf():
    raise NotImplementedError


def latcfilt():
    raise NotImplementedError


def tf2sos():
    raise NotImplementedError


def tf2ss():
    raise NotImplementedError


def tf2zpk(b, a):
    """Return zero, pole, gain (z,p,k) representation from a numerator,
    denominator representation of a linear filter.

    Convert zero-pole-gain filter parameters to transfer function form

    :param ndarray b:  numerator polynomial.
    :param ndarray a: numerator and denominator polynomials.

    :return:
        * z : ndarray        Zeros of the transfer function.
        * p : ndarray        Poles of the transfer function.
        * k : float        System gain.

    If some values of b are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    .. doctest::

        >>> import scipy.signal
        >>> from spectrum.transfer import tf2zpk
        >>> [b, a] = scipy.signal.butter(3.,.4)
        >>> z, p ,k = tf2zpk(b,a)

    .. seealso:: :func:`zpk2tf`
    .. note:: wrapper of scipy function tf2zpk
    """
    import scipy.signal
    z,p,k = scipy.signal.tf2zpk(b, a)
    return z,p,k


def ss2zpk(a,b,c,d, input=0):
    """State-space representation to zero-pole-gain representation.

    :param A: ndarray State-space representation of linear system.
    :param B: ndarray State-space representation of linear system.
    :param C: ndarray State-space representation of linear system.
    :param D: ndarray State-space representation of linear system.
    :param int input: optional For multiple-input systems, the input to use.

    :return:
        * z, p : sequence  Zeros and poles.
        * k : float System gain.

    .. note:: wrapper of scipy function ss2zpk
    """
    import scipy.signal
    z, p, k = scipy.signal.ss2zpk(a, b, c, d, input=input)
    return z, p, k


def zpk2tf(z, p, k):
    r"""Return polynomial transfer function representation from zeros and poles

    :param ndarray z: Zeros of the transfer function.
    :param ndarray p: Poles of the transfer function.
    :param float k: System gain.

    :return:
        b : ndarray Numerator polynomial.
        a : ndarray Numerator and denominator polynomials.

    :func:`zpk2tf` forms transfer function polynomials from the zeros, poles, and gains
    of a system in factored form.

    zpk2tf(z,p,k) finds a rational transfer function

    .. math:: \frac{B(s)}{A(s)} = \frac{b_1 s^{n-1}+\dots b_{n-1}s+b_n}{a_1 s^{m-1}+\dots a_{m-1}s+a_m}

    given a system in factored transfer function form

    .. math:: H(s) = \frac{Z(s)}{P(s)} = k \frac{(s-z_1)(s-z_2)\dots(s-z_m)}{(s-p_1)(s-p_2)\dots(s-p_n)}


    with p being the pole locations, and z the zero locations, with as many.
    The gains for each numerator transfer function are in vector k.
    The zeros and poles must be real or come in complex conjugate pairs.
    The polynomial denominator coefficients are returned in row vector a and
    the polynomial numerator coefficients are returned in matrix b, which has
    as many rows as there are columns of z.

    Inf values can be used as place holders in z if some columns have fewer zeros than others.

    .. note:: wrapper of scipy function zpk2tf
    """
    import scipy.signal
    b, a = scipy.signal.zpk2tf(z, p, k)
    return b, a


def zpk2ss(z, p, k):
    """Zero-pole-gain representation to state-space representation

    :param sequence z,p: Zeros and poles.
    :param float k: System gain.

    :return:
        * A, B, C, D : ndarray State-space matrices.

    .. note:: wrapper of scipy function zpk2ss
    """
    import scipy.signal
    return scipy.signal.zpk2ss(z,p,k)


