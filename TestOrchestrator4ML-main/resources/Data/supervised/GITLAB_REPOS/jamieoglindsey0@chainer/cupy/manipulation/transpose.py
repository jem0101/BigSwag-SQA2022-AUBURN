import collections

import six


def rollaxis(a, axis, start=0):
    """Moves the specified axis backwards to the given place.

    Args:
        a (cupy.ndarray): Array to move the axis.
        axis (int): The axis to move.
        start (int): The place to which the axis is moved.

    Returns:
        cupy.ndarray: A view of ``a`` that the axis is moved to ``start``.

    .. seealso:: :func:`numpy.rollaxis`

    """
    if axis >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    del tr[axis]
    tr.insert(start, axis)
    return a.transpose(*tr)


def swapaxes(a, axis1, axis2):
    """Swaps the two axes.

    Args:
        a (cupy.ndarray): Array to swap the axes.
        axis1 (int): The first axis to swap.
        axis2 (int): The second axis to swap.

    Returns:
        cupy.ndarray: A view of ``a`` that the two axes are swapped.

    .. seealso:: :func:`numpy.swapaxes`

    """
    if axis1 >= a.ndim or axis2 >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    tr[axis1], tr[axis2] = tr[axis2], tr[axis1]
    return a.transpose(*tr)


def transpose(a, axes=None):
    """Permutes the dimensions of an array.

    Args:
        a (cupy.ndarray): Array to permute the dimensions.
        axes (tuple of ints): Permutation of the dimensions. This function
            reverses the shape by default.

    Returns:
        cupy.ndarray: A view of ``a`` that the dimensions are permuted.

    .. seealso:: :func:`numpy.transpose`

    """
    if not axes:
        axes = tuple(reversed(six.moves.range(a.ndim)))
    elif len(axes) == 1 and isinstance(axes[0], collections.Iterable):
        axes = tuple(axes[0])

    if any(axis < -a.ndim or axis >= a.ndim for axis in axes):
        raise IndexError('Axes overrun')

    axes = tuple(axis % a.ndim for axis in axes)

    if list(six.moves.range(a.ndim)) != sorted(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    newarray = a.view()
    newarray._shape = tuple(a._shape[axis] for axis in axes)
    newarray._strides = tuple(a._strides[axis] for axis in axes)
    newarray._mark_dirty()
    return newarray
