import ctypes
import os

import six

from cupy import cuda

MAX_NDIM = 25


def _make_carray(n):
    class CArray(ctypes.Structure):
        _fields_ = [('data', ctypes.c_void_p),
                    ('size', ctypes.c_int),
                    ('shape', ctypes.c_int * n),
                    ('strides', ctypes.c_int * n)]
    return CArray


_carrays = [_make_carray(i) for i in six.moves.range(MAX_NDIM)]


def to_carray(data, size, shape, strides):
    global _carrays
    return _carrays[len(shape)](data, size, shape, strides)


_header_source = None


def _get_header_source():
    global _header_source
    if _header_source is None:
        header_path = os.path.join(os.path.dirname(__file__), 'carray.cuh')
        with open(header_path) as header_file:
            _header_source = header_file.read()
    return _header_source


def compile_with_cache(source, options=(), arch=None, cachd_dir=None):
    source = _get_header_source() + source
    return cuda.compile_with_cache(source, options, arch, cachd_dir)
