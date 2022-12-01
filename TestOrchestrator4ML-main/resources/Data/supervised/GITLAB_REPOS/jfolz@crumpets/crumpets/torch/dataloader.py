from __future__ import division
import uuid

# noinspection PyUnresolvedReferences
from six.moves.queue import Queue as THQueue
# noinspection PyUnresolvedReferences
from six.moves.queue import Empty
# noinspection PyUnresolvedReferences
from six.moves.queue import Full

import torch
from torch.nn import Sequential as Identity

from . import is_cpu_only
from ..dataloader import Consumer
from ..dataloader import TurboDataLoader
from ..dataloader import make_addresses
from .shm import DummyTensorManager
from .shm import SharedTensorManager
from ..broker import Pipeline
from .randomizer import Randomizer


__all__ = ['TorchTurboDataLoader']


class TorchConsumer(Consumer):
    """
    Consumer to retrieve and forward processed samples from workers.

    :param result_address:
        address to retrieve processed samples from, workers send their results to it
    :param control_address:
        address to retrieve control messages from, such as exceptions raised in other processes
    :param recv_timeout:
        time to wait in ms until another receiving attempt is made
    :param bind:
        bind addresses instead of connecting to them
    :param device:
        string or torch device that tensors are copied to
    :param gpu_augmentation:
        uses :class:`~crumpets.torch.randomizer` to gpu augment retrieved samples
    """
    def __init__(
        self,
        result_address,
        control_address,
        recv_timeout=1000,
        bind=True,
        device='cuda:0',
        gpu_augmentation=False
    ):
        if gpu_augmentation and is_cpu_only(device):
            raise ValueError(
                'cannot set gpu_augmentation=True with device %r' % device
            )
        super(TorchConsumer, self).__init__(
            result_address,
            control_address,
            recv_timeout,
            bind,
        )
        device = torch.device(device)
        self.randomizer = Randomizer().to(device) if gpu_augmentation else Identity()
        self.set_buffer_manager(DummyTensorManager(device))

    def _transform(self, data):
        if data is None:
            return None
        unpacked = super(TorchConsumer, self)._transform(data)
        return self.randomizer(unpacked)


class TorchTurboDataLoader(TurboDataLoader):
    """
    TorchTurboDataLoader is a subclass of
    :class:`~crumpets.dataloader.TurboDataLoader`
    intended for use with the Pytorch framework.
    It produces torch tensors instead of numpy arrays.

    See :class:`~crumpets.dataloader.TurboDataLoader`
    for more details on its operation.

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
    :param device:
        torch device to use,
        Defaults to 'cuda:0'.
    :param gpu_augmentation:
        Use a :class:`~crumpets.torch.randomizer.Randomizer`
        to calculate certain data augmentation operations on GPU.
        This disables said operations on the CPU side.
    """
    def __init__(self, iterable, batch_size, worker_template, nworkers,
                 length=None, num_mini_batches=1, start_iteration=0,
                 device='cuda:0', gpu_augmentation=False,
                 shared_memory=True):
        super(TorchTurboDataLoader, self).__init__(
            iterable, batch_size, worker_template, nworkers, length, num_mini_batches,
            start_iteration, shared_memory,
        )
        uid = uuid.uuid4()
        worker_addresses, consumer_addresses = make_addresses(
            uid, 'torch', (nworkers, 1))
        control_address = 'ipc://torch-control-{}.ipc'.format(uid)
        self._addresses = worker_addresses + consumer_addresses + [control_address]
        self.consumer = TorchConsumer(
            consumer_addresses[0],
            control_address,
            device=device,
            gpu_augmentation=gpu_augmentation
        )
        if shared_memory:
            nbuffers = nworkers*2 if shared_memory is True else shared_memory
            manager = SharedTensorManager(
                nbuffers,
                batch_size,
                worker_template.buffer_specs,
                device=device
            )
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
            gpu_augmentation=gpu_augmentation
        )
