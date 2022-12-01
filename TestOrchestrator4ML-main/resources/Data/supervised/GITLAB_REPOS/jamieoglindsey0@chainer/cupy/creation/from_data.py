import numpy

import cupy
from cupy import cuda
from cupy import elementwise
from cupy import internal


def array(obj, dtype=None, copy=True, ndmin=0):
    """Creates an array on the current device.

    This function currently does not support the ``order`` and ``subok``
    options.

    Args:
        obj: cupy.ndarray object or any other object that can be passed to
            :func:`numpy.array`.
        dtype: Data type specifier.
        copy (bool): If False, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        ndmin (int): Minimum number of dimensions. Ones are inserated to the
            head of the shape if needed.

    Returns:
        cupy.ndarray: An array on the current device.

    .. seealso:: :func:`numpy.array`

    """
    # TODO(beam2d): Support order and subok options
    if isinstance(obj, cupy.ndarray):
        if dtype is None:
            dtype = obj.dtype
        a = obj.astype(dtype, copy)

        ndim = a.ndim
        if ndmin > ndim:
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a
    else:
        a_cpu = numpy.array(obj, dtype=dtype, copy=False, ndmin=ndmin)
        if a_cpu.ndim > 0:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        a = cupy.ndarray(a_cpu.shape, dtype=a_cpu.dtype)
        a.data.copy_from_host(internal.get_ndarray_ptr(a_cpu), a.nbytes)
        if a_cpu.dtype == a.dtype:
            return a
        else:
            return a.view(dtype=a_cpu.dtype)


def asarray(a, dtype=None):
    """Converts an object to array.

    This function currently does not support the ``order`` option.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.

    Returns:
        cupy.ndarray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    .. seealso:: :func:`numpy.asarray`

    """
    return cupy.array(a, dtype=dtype, copy=False)


def asanyarray(a, dtype=None):
    """Converts an object to array.

    This is equivalent to cupy.asarray.

    .. seealso:: :func:`cupy.asarray`, :func:`numpy.asanyarray`

    """
    return cupy.asarray(a, dtype)


def ascontiguousarray(a, dtype=None):
    """Returns a C-contiguous array.

    Args:
        a (cupy.ndarray): Source array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: If no copy is required, it returns ``a``. Otherwise, it
        returns a copy of ``a``.

    .. seealso:: :func:`numpy.ascontiguousarray`

    """
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = numpy.dtype(dtype)

    if dtype == a.dtype and a.flags.c_contiguous:
        return a
    else:
        newarray = cupy.empty_like(a, dtype)
        elementwise.copy(a, newarray)
        return newarray


def asmatrix(data, dtype=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def copy(a):
    """Creates a copy of a given array on the current device.

    This function allocates the new array on the current device. If the given
    array is allocated on the different device, then this function tries to
    copy the contents over the devices.

    Args:
        a (cupy.ndarray): The source array.

    Returns:
        cupy.ndarray: The copy of ``a`` on the current device.

    See: :func:`numpy.copy`, :meth:`cupy.ndarray.copy`

    """
    # If the current device is different from the device of ``a``, then this
    # function allocates a new array on the current device, and copies the
    # contents over the devices.
    # TODO(beam2d): Support ordering option
    if a.size == 0:
        return cupy.empty_like(a)

    if a.data.device != cuda.Device():
        # peer copy
        a = ascontiguousarray(a)
        newarray = cupy.empty_like(a)
        newarray.data.copy_from_peer(a.data, a.nbytes)
        return newarray

    # in-device copy
    newarray = cupy.empty_like(a)
    if a.flags.c_contiguous:
        newarray.data.copy_from(a.data, a.nbytes)
    else:
        elementwise.copy(a, newarray)
    return newarray


def frombuffer(buffer, dtype=float, count=-1, offset=0):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromfile(file, dtype=float, count=-1, sep=''):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromfunction(function, shape, **kwargs):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromiter(iterable, dtype, count=-1):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromstring(string, dtype=float, count=-1, sep=''):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def loadtxt(fname, dtype=numpy.float64, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0):
    # TODO(beam2d): Implement it
    raise NotImplementedError
