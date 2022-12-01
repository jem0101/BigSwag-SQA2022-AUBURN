from cupy import reduction


def amin(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the minimum of an array or the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take the minimum.
        axis (int): Along which axis to take the minimum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amin`

    """
    return reduction.amin(
        a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the maximum of an array or the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The maximum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amax`

    """
    return reduction.amax(
        a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanmin(a, axis=None, out=None, keepdims=False):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def nanmax(a, axis=None, out=None, keepdims=False):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ptp(a, axis=None, out=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    # TODO(beam2d): Implement it
    raise NotImplementedError
