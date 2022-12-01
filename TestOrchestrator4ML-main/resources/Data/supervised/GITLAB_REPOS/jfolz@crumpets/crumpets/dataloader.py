from __future__ import division
import uuid

import os
from itertools import chain
from math import ceil
from threading import Thread
from sys import stderr

# noinspection PyUnresolvedReferences
from six.moves.queue import Queue
# noinspection PyUnresolvedReferences
from six.moves.queue import Empty
# noinspection PyUnresolvedReferences
from six.moves.queue import Full

import zmq
import msgpack

from .broker import Pipeline
from .shm import DummyBufferManager
from .shm import SharedBufferManager


def remove_files(files):
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass


def remove_ipc_handles(handles):
    remove_files([handle.replace('ipc://', '') for handle in handles])


def make_addresses(uid, pipeline, numbers=(1, 1), types=('work', 'consume')):
    addr = 'ipc://%s-%%s-%s-%%d.ipc' % (pipeline, uid)
    return [[addr % (t, n)
            for n in range(number)]
            for number, t in zip(numbers, types)]


def _check_types(vs, t):
    for v in vs:
        if not(v is None or isinstance(v, t)):
            return False
    return True


class Slicer(object):
    def __init__(self, iterable):
        if not (hasattr(iterable, '__next__') or
                hasattr(iterable, 'next')):
            iterable = iter(iterable)
        self._iterable = iterable

    def __getitem__(self, item):
        start = item.start
        stop = item.stop
        step = item.step
        if not _check_types((start, stop, step), int):
            raise ValueError('expected int or None, got: [%r:%r:%r]'
                             % (start, stop, step))
        step = step or 1
        it = self._iterable
        while start:
            next(it)
            start -= 1
            if stop:
                stop -= 1
        if stop is None:
            while 1:
                first = next(it),
                yield chain(first, (next(it) for _ in range(step-1)))
        else:
            for _ in range(stop // step):
                first = next(it),
                yield chain(first, (next(it) for _ in range(step-1)))
            rem = int(stop / step % 1 * step)
            if rem > 0:
                first = next(it),
                yield chain(first, (next(it) for _ in range(rem-1)))


class Consumer(object):
    """
    A Consumer retrieves and forward processed samples from workers.

    :param result_address:
        address to retrieve processed samples from, workers send their results to it
    :param control_address:
        address to retrieve control messages from, such as exceptions raised in other processes
    :param recv_timeout:
        time to wait in ms until another receiving attempt is made
    :param bind:
        bind addresses instead of connecting to them
    """

    def __init__(
        self,
        result_address,
        control_address,
        recv_timeout=1000,
        bind=True,
    ):
        self.running = True
        self.result = None
        self.control = None
        self.queue = None
        self.retriever = None
        self.result_address = result_address
        self.control_address = control_address
        self.recv_timeout = recv_timeout
        self.bind = bind
        self.queue_length = 1
        self.io_threads = 1
        self.buffer_manager = DummyBufferManager()

    def set_buffer_manager(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def _transform(self, data):
        if data is None:
            return None
        return self.buffer_manager.unpack(data)

    def retrieve_data(self):
        while self.running:
            try:
                return self.result.recv(copy=False)
            except zmq.Again:
                pass

    def start(self):
        """
        Starts the sample retriever thread and listen on the control stream.
        """
        self.queue = Queue(maxsize=self.queue_length)
        self.retriever = Thread(target=self._retriever_target)
        self.retriever.daemon = True
        self.retriever.start()
        self._connect_control()

    def stop(self):
        """
        Stops all threads opened by this consumer.
        """
        self.running = False
        while True:
            if self.queue is None:
                return
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
                try:
                    return msgpack.unpackb(self.control.recv(copy=False), raw=False)
                except zmq.Again:
                    pass
                item = self.queue.get()
                self.queue.task_done()
                return item
            except Empty:
                pass

    def _connect(self):
        ctx = zmq.Context(self.io_threads)
        # noinspection PyUnresolvedReferences
        self.result = ctx.socket(zmq.PULL)
        # noinspection PyUnresolvedReferences
        self.result.setsockopt(zmq.LINGER, 0)
        # noinspection PyUnresolvedReferences
        self.result.setsockopt(zmq.RCVHWM, self.queue_length)
        # noinspection PyUnresolvedReferences
        self.result.setsockopt(zmq.RCVTIMEO, self.recv_timeout)
        if self.bind:
            result_address = self.result_address.replace(
                'tcp://localhost', 'tcp://*'
            )
            self.result.bind(result_address)
        else:
            self.result.connect(self.result_address)

    def _connect_control(self):
        ctx = zmq.Context(self.io_threads)
        # noinspection PyUnresolvedReferences
        self.control = ctx.socket(zmq.SUB)
        # noinspection PyUnresolvedReferences
        self.control.setsockopt(zmq.LINGER, 0)
        # noinspection PyUnresolvedReferences
        self.control.setsockopt(zmq.RCVHWM, self.queue_length)
        # noinspection PyUnresolvedReferences
        self.control.setsockopt(zmq.RCVTIMEO, 0)
        if self.bind:
            control_address = self.control_address.replace(
                'tcp://localhost', 'tcp://*'
            )
            self.control.bind(control_address)
        else:
            self.control.connect(self.control_address)
        import time
        time.sleep(1)
        # noinspection PyUnresolvedReferences
        self.control.setsockopt(zmq.SUBSCRIBE, bytes('', 'utf-8'))

    def _retriever_target(self):
        self._connect()
        try:
            while self.running:
                item = self._transform(self.retrieve_data())
                self.queue.put(item)
                self.queue.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


class TurboDataLoader(object):
    """
    TurboDataLoader provides fast parallel loading and processing of input data.
    Use :class:`~crumpets.torch.dataloader.TorchTurboDataLoader`
    for a version supporting gpu and pytorch tensors.

    Always use the loader inside of a with statement,
    otherwise workers and consumer won't start and stop.

    `TurboDataLoader`s are intended to be used as iterators.
    Each iteration yields the following data structure:

    .. code-block:: python
        (
            iteration,
            [ # list with 1 item per mini-batch
                { ... } # sample_dict
            ].
        )

    By default `iteration` starts at 0 and counts the number of
    batches that the loader has yielded.
    The list contains as many mini-batches as specified by
    `num_mini_batches`.
    Note that the number of samples across all mini-batches
    is equal to `batch_size`,
    i.e., `batch_size` must be divisible by `num_mini_batches`.
    Finally each mini-batch is a dictionary that contains
    key-value-pairs produced by the workers.
    E.g., a :class:`~crumpets.workers.ClassificationWorker`
    produces keys `'image'`, `'label'`, and `'augmentation'`.
    Image and label are arrays and augmentation contains a list
    of one dictionary per sample in the batch with parameters
    used to create said sample.

    Example usage:

    .. code-block:: python

        model = make_some_model()
        with loader:
            for epoch in range(epochs):
                for iteration, mini_batch in loader:
                    for sample in mini_batch:
                        sample = model(sample)
                        images = sample['image']
                        ...

    Depending on parameters, the TurboDataLoaders starts several processes,
    some of which cannot be started with the standard
    "fork" method that Python uses in *nix systems.
    This can result in crashing with an obscure error message.
    Thus loaders need to be guarded against starting in non-main modules, i.e.:

    .. code-block:: python

        if __name__ == "__main__":
            # stuff
            with loader:
                # other stuff

    :param iterable:
        An iterable providing a sample per iteration.
    :param batch_size:
        The amount of samples per batch.
    :param worker_template:
        An actual worker instance, determines the  kind of processing.
        Has to inherit crumpets.broker.Worker.
    :param nworkers:
        Number of workers processing the samples simultaneously.
        worker_template is copied to create them.
    :param length:
        Specifies the length of the dataset.
        Defaults to the actual length of iterable (if available).
        If given differs from default,
        the number of iterations per epoch is modified accordingly.
    :param num_mini_batches:
        Number of mini_batches per batch.
    :param start_iteration:
        Start the iteration counter from this number.
        Useful when resuming training.
    :param shared_memory:
        Whether to use shared memory to transfer data from workers.
        If 0 or `False`, shared memory is disabled.
        If `True`, `2*nworkers` shared buffers will be used.
        If any number > 0, that number of buffers will be used.
        A value of 1 is strongly discouraged to prevent deadlocks.
        Permanently storing values returned by a loader may also
        cause deadlocks.
    """
    def __init__(self, iterable, batch_size, worker_template, nworkers,
                 length=None, num_mini_batches=1, start_iteration=0,
                 shared_memory=True):
        uid = uuid.uuid4()
        worker_addresses, consumer_addresses = make_addresses(
            uid, 'torch', (nworkers, 1))
        control_address = 'ipc://torch-control-{}.ipc'.format(uid)
        self._addresses = worker_addresses + consumer_addresses + [control_address]

        if batch_size / num_mini_batches != batch_size // num_mini_batches:
            print('batch_size %d and num_mini_batches %d don\'t match'
                  % (batch_size, num_mini_batches))

        self.num_mini_batches = num_mini_batches
        self.batch_size = batch_size
        self.mini_batch_size = batch_size // num_mini_batches
        self.nworkers = nworkers
        self.iterations = start_iteration
        self.epoch_iterations = 0
        self.length = 0
        if length is None:
            try:
                length = len(iterable)
            except (TypeError, AttributeError):
                pass
        if length is not None:
            self.set_length(length)

        self.consumer = Consumer(
            consumer_addresses[0],
            control_address,
        )
        if shared_memory:
            nbuffers = nworkers*2 if shared_memory is True else shared_memory
            manager = SharedBufferManager(nbuffers, batch_size,
                                          worker_template.buffer_specs)
            worker_template.set_buffer_manager(manager)
            self.consumer.set_buffer_manager(manager)
        self.pipeline = Pipeline(
            worker_template,
            nworkers,
            iterable,
            self.mini_batch_size,
            worker_addresses,
            consumer_addresses,
            control_address=control_address,
            gpu_augmentation=False
        )

    def set_length(self, length):
        """
        Set the length of enclosed iterable.
        Modifies epoch_iterations accordingly.
        :param length: len(iterable)
        """
        self.length = length
        self.epoch_iterations = int(ceil(length / self.batch_size))

    def set_epoch_iterations(self, iterations):
        """
        Set number of iterations in one epoch.
        Does not modify length.
        :param iterations: number of iterations per epoch
        """
        self.epoch_iterations = iterations

    def start(self):
        """
        Start the processing pipeline.
        """
        self.consumer.start()
        self.pipeline.start()

    def __enter__(self):
        self.start()
        return self

    def stop(self):
        """
        Stop the processing pipeline.
        """
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if hasattr(self, 'consumer'):
            self.consumer.stop()
        remove_ipc_handles(self._addresses)

    __del__ = stop

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __consume__(self):
        """
        Generator for (iteration, sample) pairs
        :return:
        """
        while 1:
            item = self.consumer.retrieve()
            if item is None:
                raise StopIteration()
            if "is_control_msg" in item and item['is_control_msg']:
                if "raise_error" in item and item["raise_error"]:
                    self.stop()
                    if "traceback" in item:
                        print("\n", *item['traceback'], file=stderr)
                        raise IOError(item["msg"])
                    else:
                        raise IOError(item["msg"])
                else:
                    print("[CONTROL MSG] {}".format(item["msg"]))
            yield item

    def __iter__(self):
        """
        Generator for (iteration, sample) pairs
        :return:
        """
        gen = Slicer(self.__consume__())
        m = self.num_mini_batches
        if self.epoch_iterations > 0:
            end = self.epoch_iterations * self.num_mini_batches
            gen = gen[:end:m]
        for batch in gen:
            self.iterations += 1
            yield self.iterations, batch
