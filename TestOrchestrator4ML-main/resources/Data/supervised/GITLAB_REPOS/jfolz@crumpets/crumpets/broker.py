from __future__ import print_function

import time
import traceback
from itertools import cycle
from threading import Thread
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
import multiprocessing as mp
from collections import defaultdict

from six import with_metaclass
from six.moves.queue import Queue
from six.moves.queue import Empty
from six.moves.queue import Full

import numpy as np
import zmq
from msgpack import packb
from msgpack import unpackb
from msgpack_numpy import encode
from msgpack_numpy import decode

from . import procname


class ProducerBase(with_metaclass(ABCMeta, mp.Process)):
    """
    Abstract base class for producer processes.
    Producers are the first stage of the pre-processing pipeline that
    load data into memory and supply it to workers.
    Implement the yield_requests method to customize its behavior.

    :param work_addresses:
        List of worker addresses the producer pushes work to;
        cycled through for load balancing
    :param daemon:
        Flag whether this Producer is a daemon process;
        see multiprocessing.Process
    :param queue_length:
        Length of send queue per worker socket
    :param io_threads:
        Number of IO threads to use; 1 is fine for almost all cases
    """
    def __init__(self, work_addresses, daemon=True, queue_length=8, io_threads=1):
        mp.Process.__init__(self, name='pyProducer')
        self.work_addresses = work_addresses
        self.queue_length = queue_length
        self.io_threads = io_threads
        self.running = mp.Value('b')
        self.running.value = True
        self.daemon = daemon

    def stop(self):
        self.running.value = False

    # noinspection PyUnresolvedReferences
    def run(self):
        procname.setprocname('pyProducer')
        ctx = zmq.Context(io_threads=self.io_threads)
        worker = []
        for work_address in self.work_addresses:
            work = ctx.socket(zmq.PUSH)
            work.setsockopt(zmq.SNDHWM, self.queue_length)
            work.bind(work_address.replace('tcp://localhost', 'tcp://*'))
            worker.append(work)
        worker = cycle(worker)
        gen = self.yield_requests()
        while self.running.value:
            try:
                next(worker).send_multipart(next(gen))
            except (KeyboardInterrupt, StopIteration):
                self.stop()

    def yield_requests(self):
        raise NotImplementedError('implement yield_requests as generator')


class Producer(ProducerBase):
    """
    Producer implementation that reads sequentially from arbitrary
    iterable objects. Items must be a msgpack messages that are
    understood by the workers.
    :param iterable:
        iterable of msgpack messages
    :param batch:
        batch size for workers
    """
    def __init__(self, work_addresses, iterable, batch,
                 queue_length=8, io_threads=1):
        ProducerBase.__init__(self, work_addresses,
                              queue_length=queue_length,
                              io_threads=io_threads)
        if not hasattr(iterable, '__next__'):
            iterable = iter(iterable)
        self.iterable = iterable
        self._batch = range(batch)

    def yield_requests(self):
        request = tuple(next(self.iterable) for _ in self._batch)
        while request:
            yield request
            request = tuple(next(self.iterable) for _ in self._batch)


class ConsumerBase(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for Consumers, the final pipeline stage.
    Implement the _transform method to define subclass behavior.
    """
    # noinspection PyUnresolvedReferences
    def __init__(
            self,
            result_address,
            recv_timeout=1000,
            queue_length=3,
            bind=True,
            io_threads=1
    ):
        self.running = True
        self.ctx = zmq.Context(io_threads)
        self.result = self.ctx.socket(zmq.PULL)
        self.result.setsockopt(zmq.LINGER, 0)
        self.result.setsockopt(zmq.RCVHWM, queue_length)
        self.result.setsockopt(zmq.RCVTIMEO, recv_timeout)
        if bind:
            result_address = result_address.replace('tcp://localhost', 'tcp://*')
            self.result.bind(result_address)
        else:
            self.result.connect(result_address)

    def stop(self):
        self.running = False

    def retrieve(self):
        return self._transform(self.retrieve_data())

    @abstractmethod
    def _transform(self, data):
        """
        Implement this function to transform raw data received from workers
        to a usable Python object.

        :param data:
            raw bytes from workers
        :return:
            transformed Python object
        """
        pass

    def retrieve_data(self):
        while self.running:
            try:
                return self.result.recv(copy=False)
            except zmq.Again:
                pass


class ThreadedConsumerBase(with_metaclass(ABCMeta, ConsumerBase)):
    """
    Abstract base class for Consumers that receive and transform data
    on a separate thread.
    Implement the _transform method to define subclass behavior.
    """
    def __init__(
        self,
        result_address,
        recv_timeout=2000,
        queue_length=3,
        bind=True,
        io_threads=1
    ):
        ConsumerBase.__init__(
            self,
            result_address,
            recv_timeout,
            queue_length,
            bind,
            io_threads
        )
        self.queue = Queue(maxsize=queue_length)
        self.retriever = Thread(target=self.__retriever_target)
        self.retriever.daemon = True
        self.retriever.start()

    def stop(self):
        ConsumerBase.stop(self)
        while True:
            try:
                self.queue.get(False)
            except Empty:
                pass
            try:
                self.queue.put(None, timeout=1)
                break
            except Full:
                pass

    def retrieve(self):
        while self.running:
            try:
                return self.queue.get(timeout=0.01) or None
            except Empty:
                pass

    def __retriever_target(self):
        while self.running:
            self.queue.put(self._transform(self.retrieve_data()))
        self.queue.put(None)


class Consumer(ThreadedConsumerBase):
    """
    Basic threaded Consumer that receives und unpacks msgpack messages.
    """
    def _transform(self, data):
        return unpack(data)


class Proxy(mp.Process):
    """
    Utility class that receives and redirects zmq PULL/PUSH streams.
    """
    def __init__(
            self,
            in_address,
            out_address,
            queue_length=1,
            daemon=True,
    ):
        mp.Process.__init__(self, name='pyProxy')
        self.in_address = in_address.replace('tcp://localhost', 'tcp://*')
        self.out_address = out_address.replace('tcp://localhost', 'tcp://*')
        self.queue_length = queue_length
        self.daemon = daemon

    # noinspection PyUnresolvedReferences
    def run(self):
        procname.setprocname('pyProxy')
        ctx = zmq.Context(1)
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVHWM, 1)
        pull.bind(self.in_address)
        push = ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, self.queue_length)
        push.bind(self.out_address)
        try:
            zmq.proxy(pull, push)
        except KeyboardInterrupt:
            pass
        finally:
            pull.close()
            push.close()
            ctx.term()


class Value(object):
    def __init__(self, *_, **__):
        self.value = None


class Worker(with_metaclass(ABCMeta)):
    """
    Abstract base class for workers.
    Implement the process method to define the behavior of subclasses.

    .. note::
        set_addresses must be called before starting a worker.
        The :class:`~crumpets.dataloader.TurboDataLoader` does this for you.

    :param timeout:
        zmq socket timeout in milliseconds
    :param daemon:
        set daemon flag - used in process
    :param gpu_augmentation:
        set GPU augmentation flag
    """
    def __init__(self, timeout=1000, daemon=True, gpu_augmentation=False):
        # mp.Process.__init__(self, name='pyWorker')
        self.work_address = None
        self.result_address = None
        self.control_address = None
        self.timeout = timeout
        self.running = Value()  # replaced with multiprocessing.Value
        self.running.value = True
        self.daemon = daemon
        self.gpu_augmentation = gpu_augmentation

    def set_addresses(self, work, result, control):
        """
        Set all required zmq addresses.
        Required before run can be invoked.

        :param work:
            address where work is received on
        :param result:
            results are pushed to this address
        :param control:
            control message are sent here, e.g.,
            exceptions that occurred while processing
        """
        self.work_address = work
        self.result_address = result
        self.control_address = control

    def set_gpu_augmentation(self, val):
        """
        Sets the gpu_augmentation flag to given value, true disables
        all cpu_augmentations for which a gpu version is available.
        Note that this does not directly activate usage of gpu augmentation, as for that
        a :class:`~crumpets.torch.randomizer` module is used, which usually
        the :class:`~crumpets.dataloader.TurboDataLoader` takes care of.

        :param val:
            boolean flag
        """
        self.gpu_augmentation = val

    def stop(self):
        """
        Stops the worker process.
        """
        self.running.value = False

    # noinspection PyUnresolvedReferences
    def inner(self):
        addr = self.work_address, self.result_address, self.control_address
        if not all(addr):
            raise RuntimeError(
                'Some addresses not set. '
                'Call set_addresses before start. '
                'work: %r, result: %r, control: %r' % addr
            )
        procname.setprocname('pyWorker')
        ctx = zmq.Context()
        work = ctx.socket(zmq.PULL)
        work.setsockopt(zmq.RCVHWM, 1)
        work.setsockopt(zmq.RCVTIMEO, self.timeout)
        work.setsockopt(zmq.LINGER, 0)
        work.connect(self.work_address)
        result = ctx.socket(zmq.PUSH)
        result.setsockopt(zmq.SNDHWM, 1)
        result.setsockopt(zmq.LINGER, 0)
        result.connect(self.result_address)
        while self.running.value:
            try:
                request = work.recv_multipart(copy=False)
                if request[0] == b'\x00':
                    self.stop()
                    break
                for data in self.process(request):
                    if data:
                        result.send(data)
            except zmq.Again:
                pass

    def run(self):
        """
        Starts the worker process.
        """
        try:
            self.inner()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            trace = traceback.format_exception(type(e), e, e.__traceback__)
            self._send_control_msg(str(e), trace)
            self.stop()
            raise

    @abstractmethod
    def process(self, data):
        """
        Implement this method to define worker behavior.
        Can return an iterable to create several batches from one input.
        This method can return an iterable or define a generator
        with the yield keyword.
        For instance: :func:`~crumpets.broker.BufferWorker.process`
.
        :param data:
            multipart zmq message from Producer to process
        :return:
            iterable of zmq messages to send to Consumer
        """
        return []

    # noinspection PyUnresolvedReferences
    def _send_control_msg(self, msg, trace=None):
        if not self.control_address:
            raise ValueError("no control address specified, "
                             "control msg to send was: {}".format(msg))
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.connect(self.control_address)
        # TODO sync that instead of sleep?
        # Dunno if that's possible in zmq - Joachim
        time.sleep(1)
        socket.send(packb({
            "msg": msg,
            "is_control_msg": True,
            "raise_error": True,
            "traceback": trace
        }, use_bin_type=True))
        # TODO may need to sleep some before the context dies


def unpack(obj):
    return unpackb(obj, object_hook=decode, raw=False)


def make_fill_value(shape, dtype, fill_value=0):
    """
    Create a numpy array for a given fill value.
    This array can be used to fill any array of the given shape and dtype,
    e.g., arr[:] = make_fill_value(arr.shape, arr.dtype, 17) will set all
    elements of arr to 17.

    Note: An implicit first dimension for the batch size is added.

    fill_value can be a scalar or iterable.
    Iterables are padded ith unit dimensions until they match the number
    of dimensions of the given shape, e.g.:

    >>> make_fill_value((3, 224, 224), np.uint8, (1, 2, 3))
    array([[[[1]], [[2]], [[3]]]], dtype=uint8)

    The resulting fill value array has shape (1, 3, 1, 1).

    :param shape:
        array shape
    :param dtype:
        array dtype
    :param fill_value:
        optional fill value(s)
    :return:
        fill value array
    """
    fill_value = np.asarray(fill_value, dtype)
    if len(fill_value.shape) == 0:
        shape = (1,) + (1,) * len(shape)
    else:
        filler_dims = len(shape) - len(fill_value.shape)
        shape = (1,) + fill_value.shape + (1,) * filler_dims
        # noinspection PyArgumentList
    return fill_value.reshape(shape)


def make_bufferspec(buf):
    """
    Turn numpy.ndarray into buffer specification:
    :param buf:
        np.ndarray or buffer spec
    :return:
        tuple(shape, dtype, fill_value)
    """
    if isinstance(buf, np.ndarray):
        shape = buf.shape
        # Check for variance in buffer dimensions:
        # If dimension is unit, use slice(None)
        # If var > 0, slice(None) is equivalent to array[:].
        # If var == 0, use index 0 element as representative.
        ind = tuple(slice(None) if s == 1 or np.any(buf.var(d) > 0) else 0
                    for d, s in enumerate(shape))
        fill_value = buf[ind]
        if np.isscalar(fill_value):
            fill_value = fill_value.item()
        return shape, buf.dtype, fill_value
    else:
        return buf[0], buf[1], make_fill_value(*buf)


def make_buffer(batchsize, shape, dtype, fill_value):
    """
    Create an array for a given batch size and buffer spec.
    Resulting array has shape = (batchsize,) + shape.

    :param batchsize:
        size of the first dimension
    :param shape:
        remaining shape of the array
    :param dtype:
        numpy dtype of the array
    :param fill_value:
        array comes pre-filled with this value
    :return:
        array
    """
    buf = np.empty((batchsize,) + tuple(shape), dtype)
    buf[:] = fill_value
    return buf


class BufferManager(object):
    """
    BufferManager is a compatibility class that replaces the SharedDictManager
    for cases where shared memory is not used by the pipeline.
    It creates buffers from buffer specs for use with the BufferWorker.
    """
    def __init__(self, batch_size, buffer_specs):
        self.batch_size = batch_size
        self.buffer_specs = buffer_specs
        self.buffers = None

    def next(self):
        """
        Return the dictionary of buffers as defined by buffer specs.

        :return:
            buffer dictionary
        """
        if self.buffers is None:
            if not self.buffer_specs:
                return {}
            n = self.batch_size
            self.buffers = {
                k: make_buffer(n, *spec)
                for k, spec in self.buffer_specs.items()
            }
        return self.buffers

    @staticmethod
    def pack(obj):
        """
        Pack an object using msgpack.
        Any shared object are replaced by references.

        :param obj:
            object to pack
        :return:
            msgpack message bytes
        """
        return packb(obj, use_bin_type=True, default=encode)

    @staticmethod
    def unpack(data):
        """
        Unpack an msgpack message.
        Any shared object references are replaced with the object.

        :param data:
            msgpack message bytes
        :return:
            packed objects
        """
        return unpackb(data, object_hook=decode, raw=False)


class BufferWorker(with_metaclass(ABCMeta, Worker)):
    """
    Base class for workers that use constant-size buffers.

    :param buffer_manager:
        Dict of buffer specs (shape, dtype, fill_value).
        fill_value is optional and defaults to 0.
        It must be either a scalar or iterable of length equal to
        the number of channels in the respective image.
    :param param_groups:
        Dict of fixed parameter dicts.
        To be used in conjunction with buffers of the same key.
    :param kwargs:
        Passed to broker.Worker.
    """
    def __init__(self, buffer_manager=None, **kwargs):
        Worker.__init__(self, **kwargs)
        self.buffer_manager = buffer_manager
        self.buffer_specs = {}
        self.fill_values = {}
        self.params = {}

    def get_buffer_manager(self):
        """
        Returns the current buffer manager.
        May be None.
        :return: `BufferManager` or `SharedBufferManager` object
        """
        return self.buffer_manager

    def set_buffer_manager(self, buffer_manager):
        """
        Set the buffer manager to be used by this worker.
        Can be None, in which case a `BufferManager` will
        be created as necessary.

        :param buffer_manager:
            a `BufferManager` or `SharedBufferManager` object, or None
        """
        self.buffer_manager = buffer_manager

    def add_buffer(self, key, buf):
        """
        Register a new buffer with the worker.

        :param key:
            name of the buffer
        :param buf:
            buffer spec or array to use as template
        """
        spec = make_bufferspec(buf)
        self.buffer_specs[key] = spec
        self.fill_values[key] = spec[2]

    def add_params(self, key, params, default=None):
        """
        Add a parameter group to the worker.

        :param key:
            name of the parameters
        :param params:
            parameter object, usually dictionary
        :param default:
            default value to use if params is None
        """
        self.params[key] = default if params is None else params

    @abstractmethod
    def prepare(self, sample, batch, buffers):
        """
        Implement this method to define the behavior of the BufferWorker
        subclass. Results must be written to buffers and/or batch object.

        :param sample:
            individual sample object to process
        :param batch:
            the object the sample belongs to;
            append values to lists as necessary
        :param buffers:
            output buffers to use for this sample
        """
        pass

    def process(self, request):
        n = len(request)
        if self.buffer_manager is None:
            self.buffer_manager = BufferManager(n, self.buffer_specs)
        buffers = self.buffer_manager.next()
        rows = [{k: buf[i] for k, buf in buffers.items()}
                for i in range(n)]
        batch = defaultdict(list)
        for row, sample in zip(rows, request):
            try:
                self.prepare(self.buffer_manager.unpack(sample), batch, row)
            except ValueError as e:
                for k, buf in row.items():
                    buf[...] = self.fill_values[k]
                print('[BufferWorker] cannot prepare sample:', str(e))
        # request is smaller than batch, zero remaining rows
        if n < len(rows):
            for k in buffers:
                buffers[k][n:] = self.fill_values[k]

        batch.update(buffers)
        return self.buffer_manager.pack(batch),  # comma to return tuple


class Dispatcher(object):
    """
    The Dispatcher creates worker processes from a worker template,
    can starts and stops them and monitor their status.

    :param worker_template:
        instance of Worker subclass to use as template for workers;
        copy.copy is used to create as many objects as needed
    :param nworkers:
        number of worker processes to start
    :param work_addresses:
        list of work addresses to use; cycled through
    :param result_addresses:
        list of result addresses to use; cycles through
    :param control_address:
        control address workers can send status updates on
    :param daemon:
        daemon flag for processes, see multiprocessing.Process
    :param gpu_augmentation:
        bool passed to workers, true disables cpu augmentations
        where gpu versions are available in :class:`~crumpets.torch.randomizer`;
        if None worker_template.gpu_augmentation is used
    """
    def __init__(
            self,
            worker_template, nworkers,
            work_addresses, result_addresses, control_address,
            daemon=None,
            gpu_augmentation=None
    ):
        work_address = cycle(work_addresses)
        result_address = cycle(result_addresses)
        try:
            buffer_manager = worker_template.get_buffer_manager()
            worker_template.set_buffer_manager(None)
        except AttributeError:
            buffer_manager = None
        self.workers = [deepcopy(worker_template) for _ in range(nworkers)]
        for worker, work, result \
                in zip(self.workers, work_address, result_address):
            worker.running = mp.Value('b')
            worker.running.value = worker_template.running.value
            worker.set_addresses(work, result, control_address)
            worker.set_gpu_augmentation(gpu_augmentation or worker.gpu_augmentation)
            try:
                worker.set_buffer_manager(buffer_manager)
            except AttributeError:
                pass
        try:
            worker_template.set_buffer_manager(buffer_manager)
        except AttributeError:
            pass
        self.processes = [mp.Process(target=w.run) for w in self.workers]
        for w in self.processes:
            w.daemon = daemon or w.daemon

    def start(self):
        for proc in self.processes:
            proc.start()

    def stop(self):
        for worker in self.workers:
            worker.stop()

    def active(self):
        """
        True if any workers are alive.
        """
        return any([worker.is_alive() for worker in self.workers])

    def terminate(self):
        for proc in self.processes:
            proc.terminate()


class Pipeline(object):
    def __init__(
            self,
            worker_template, nworkers, iterable, batch_size,
            work_addresses, result_addresses,
            producer_kwargs=None, control_address=None,
            gpu_augmentation=None
    ):
        self.dispatcher = Dispatcher(worker_template, nworkers,
                                     work_addresses, result_addresses, control_address,
                                     gpu_augmentation=gpu_augmentation)
        producer_kwargs = producer_kwargs or {}
        self.producer = Producer(
            work_addresses,
            iterable,
            batch_size,
            **producer_kwargs
        )

    def start(self):
        self.dispatcher.start()
        self.producer.start()

    def stop(self):
        try:
            self.producer.stop()
            self.producer.terminate()
        except AttributeError:
            pass  # producer already terminated/does not exist
        try:
            self.dispatcher.stop()
            self.dispatcher.terminate()
        except AttributeError:
            pass  # dispatcher already terminated/does not exist

    __del__ = stop
