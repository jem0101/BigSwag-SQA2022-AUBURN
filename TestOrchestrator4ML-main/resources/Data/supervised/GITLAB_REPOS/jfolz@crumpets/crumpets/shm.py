from operator import mul
from functools import reduce
from math import ceil
from ctypes import c_uint64
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import weakref
import warnings

import numpy as np
from msgpack import Packer
from msgpack import packb
from msgpack import unpackb
from msgpack import ExtType
from msgpack_numpy import encode as _encode_numpy
from msgpack_numpy import decode as _decode_numpy


__all__ = [
    'shared_array',
    'DummyBufferManager',
    'SharedBufferManager',
]


def make_packer(default=None):
    return PicklablePacker(use_bin_type=True, default=default)


class PicklablePacker(Packer):
    def __init__(self, *args, **kwargs):
        Packer.__init__(self, *args, **kwargs)
        self.default_fun = kwargs.get('default')

    def __reduce__(self):
        return make_packer, (self.default_fun,)


def shared_array(shape, dtype=np.float32):
    """
    Create a numpy array that resides in shared memory.
    Memory is aligned to 8 bytes.

    :param shape:
        array shape
    :param dtype:
        numpy dtype
    :return: np.ndarray
    """
    size = reduce(mul, shape, 1)
    itemsize = np.dtype(dtype).itemsize
    alloc = RawArray(c_uint64, int(ceil(size * itemsize / 8)))
    return np.frombuffer(alloc, dtype, size).reshape(shape)


EXT_SHARED = ord('s')


class DummyBufferManager(object):
    """
    Dummy replacement for SharedBufferManager.
    Supports pack and unpack, but not next methods.
    """
    def __init__(self):
        self._packer = make_packer()

    def next(self):
        raise NotImplementedError('DummyBufferManager does not support next')

    def pack(self, obj):
        """
        Pack an object using msgpack.
        :param obj: object to pack
        :return: msgpack message bytes
        """
        return self._packer.pack(obj)

    def unpack(self, data):
        """
        Unpack an msgpack message.
        :param data: msgpack message bytes
        :return: packed objects
        """
        return unpackb(data, object_hook=_decode_numpy, raw=False)


class SharedBufferManager(object):
    """
    SharedBufferManager allows transparent sharing of memory between processes.
    On creation the specified number of shared memory buffers are created
    according to batch size and buffer specs.

    `next` returns dict of numpy arrays
    that point to a set of shared memory buffers.
    `next` blocks until as set of buffers becomes available.
    If more than one buffer spec is given, next will always return one buffer
    for each spec and will only reuse a set of buffers when none of them are
    in use.

    pack serializes an arbitrary python object to msgpack format.
    It detects shared buffers and replaces them with a "pointer"
    as extension type `EXT_SHARED`.
    This makes packing fast and independent of array size.

    unpack detects "pointer" and replaces them with the shared buffer.

    Usage:

    * Sender calls next to get a set ob available buffers.
    * Sender modifies buffers, calls pack and sends message to receiver.
    * Receiver receives the message and calls unpack.
    * Receiver uses the unpacked arrays and ensures that they are deleted
      at some point, either by going out of scope or explicitly deleting them.
      Storing shared buffers permanently may cause deadlocks.
    """
    def __init__(self, num_buffers, batch_size, buffer_specs, _queueclass=mp.Queue):
        if num_buffers == 1:
            warnings.warn('[SharedBufferManager] num_buffers=1, this may produce deadlocks')
        self.batch_size = batch_size
        self.num_buffers = num_buffers
        self.buffer_specs = {
            k: ((batch_size,) + tuple(spec[0]), spec[1])
            for k, spec in buffer_specs.items()
        }
        self._buffer_sets = self._create_buffers()
        self._in_use = [{} for _ in self._buffer_sets]
        # self._buffers = {id(o): o for d in self._buffer_sets for o in d.values()}
        self._alias = {}
        self._available = _queueclass()
        for i, _ in enumerate(self._buffer_sets):
            self._available.put(i)
        self._packer = make_packer(self._encode)

    def _create_buffers(self):
        return [
            {k: (i, spec, shared_array(spec[0], spec[1]))
             for k, spec in self.buffer_specs.items()}
            for i in range(self.num_buffers)
        ]

    def _signal_done(self, i, k):
        self._in_use[i].pop(k)
        if not self._in_use[i]:
            self._available.put(i)

    def _encode(self, obj):
        try:
            i = self._alias[id(obj)]
            return ExtType(EXT_SHARED, packb(i))
        except ValueError:
            return _encode_numpy(obj)

    @staticmethod
    def _create_alias_decode(obj):
        return obj.reshape(obj.shape)

    def _decode(self, code, data):
        if code == EXT_SHARED:
            i, k = unpackb(data, raw=False)
            i, spec, alloc = self._buffer_sets[i][k]
            self._in_use[i][k] = True
            array = self._create_alias_decode(alloc)
            weakref.finalize(array, self._signal_done, i, k)
            return array
        else:
            return ExtType(code, data)

    def close(self):
        """
        Close the queue and unblock any processes waiting on next.
        """
        self._available.put(None)
        self._available.close()

    @staticmethod
    def _create_alias_next(obj):
        return obj.reshape(obj.shape)

    def next(self):
        i = self._available.get()
        if i is None:
            self.close()
            return
        allocs = self._buffer_sets[i]
        buffers = {k: self._create_alias_next(alloc)
                   for k, (_, spec, alloc) in allocs.items()}
        for k in allocs:
            self._alias[id(buffers[k])] = i, k
        return buffers

    def pack(self, obj):
        """
        Pack an object using msgpack.
        Any shared object are replaced by references.
        :param obj: object to pack
        :return: msgpack message bytes
        """
        return self._packer.pack(obj)

    def unpack(self, data):
        """
        Unpack an msgpack message.
        Any shared object references are replaced with the object.
        :param data: msgpack message bytes
        :return: packed objects
        """
        return unpackb(data, object_hook=_decode_numpy,
                       ext_hook=self._decode, raw=False)
