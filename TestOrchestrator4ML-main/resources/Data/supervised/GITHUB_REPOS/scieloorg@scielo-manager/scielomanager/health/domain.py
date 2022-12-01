#coding: utf-8
import logging

import zerorpc

import health


logger = logging.getLogger(__name__)


class BackendUnavailable(Exception):
    """ Backend de verificação de saúde está indisponível.
    """


class StatusChecker(object):
    """ Realiza e agrupa as verificações de saúde do sistema.
    """

    def __init__(self):
        self.Client = health.Client
        self.client = None

    def __enter__(self, **kwargs):
        self.dial(**kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def dial(self, timeout=5):
        self.client = self.Client(timeout=timeout)

    def close(self):
        try:
            self.client.close()
        except AttributeError as exc:
            logger.exception(exc)

        return None

    @property
    def is_fully_operational(self):
        return all([st['status'] for st in self.overall_status().values()])

    def overall_status(self):
        if not hasattr(self, '_overall_status'):
            try:
                setattr(self, '_overall_status', self.client.overall_status())
            except (zerorpc.LostRemote, zerorpc.TimeoutExpired) as e:
                logger.exception(e)
                raise BackendUnavailable(e.message)

        return self._overall_status

    @property
    def elapsed_time(self):
        try:
            return self.client.elapsed_time()
        except (zerorpc.LostRemote, zerorpc.TimeoutExpired) as e:
            logger.exception(e)
            raise BackendUnavailable(e.message)

