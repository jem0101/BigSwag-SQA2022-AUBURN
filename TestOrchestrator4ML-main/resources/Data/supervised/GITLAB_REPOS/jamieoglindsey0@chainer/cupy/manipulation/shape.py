import collections

import numpy

from cupy import internal


def reshape(a, newshape):
    """Returns an array with new shape and same elements.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        newshape (int or tuple of ints): The new shape of the array to return.
            If it is an integer, then it is treated as a tuple of length one.
            It should be compatible with ``a.size``. One element can be -1,
            which is automatically replaced with the appropriate value to make
            the shape compatible with ``a.size``.

    Returns:
        cupy.ndarray: A reshaped view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.reshape`

    """
    # TODO(beam2d): Support ordering option
    if numpy.isscalar(newshape):
        newshape = newshape,
    newshape = internal.infer_unknown_dimension(newshape, a.size)
    if len(newshape) == 1 and \
       isinstance(newshape[0], collections.Iterable):
        newshape = tuple(newshape[0])

    size = a.size
    if internal.prod(newshape) != size:
        raise RuntimeError('Total size mismatch on reshape')

    newstrides = internal.get_strides_for_nocopy_reshape(a, newshape)
    if newstrides is not None:
        newarray = a.view()
    else:
        newarray = a.copy()
        newstrides = internal.get_strides_for_nocopy_reshape(
            newarray, newshape)
    newarray._shape = newshape
    newarray._strides = newstrides
    newarray._mark_f_dirty()
    return newarray


def ravel(a):
    """Returns a flattend array.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be flattened.

    Returns:
        cupy.ndarray: A flattened view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.ravel`

    """
    # TODO(beam2d): Support ordering option
    return reshape(a, -1)
