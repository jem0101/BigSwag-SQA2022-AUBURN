from __future__ import print_function, division, absolute_import

import json
import logging
import tqdm
import sys
from collections import OrderedDict
from datetime import datetime


def get_logfilename(
        prefix='',
        dateformat='%Y-%m-%dt%H-%M-%S',
        pathformat='%s%s.log'
):
    s = datetime.strftime(datetime.now(), dateformat)
    return pathformat % (prefix, s)


class JSONLogger(logging.Logger):
    """
    A subclass of the default Python Logger that uses
    the JSONLines output format.
    """
    def __init__(self, name, filename, level=logging.NOTSET):
        logging.Logger.__init__(self, name, level)
        handler = logging.FileHandler(filename, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def _msg(self, kwargs):
        d = OrderedDict(timestamp=datetime.utcnow().isoformat() + '+0')
        d.update(kwargs)
        return json.dumps(d)

    def debug(self, **kwargs):
        logging.Logger.debug(self, self._msg(kwargs))

    def info(self, **kwargs):
        logging.Logger.info(self, self._msg(kwargs))

    def warning(self, **kwargs):
        logging.Logger.warning(self, self._msg(kwargs))

    def error(self, **kwargs):
        logging.Logger.error(self, self._msg(kwargs))

    def exception(self, exc_info=True, **kwargs):
        logging.Logger.error(self, self._msg(kwargs),
                             exc_info=exc_info)

    def critical(self, **kwargs):
        logging.Logger.critical(self, self._msg(kwargs))

    fatal = critical

    def log(self, level, **kwargs):
        logging.Logger.log(self, level, self._msg(kwargs))


class SilentLogger(object):
    """
    Replacement logger for a logger that does not log anything.
    Useful when running multiple processes, but not all of them
    should log results.
    """
    def debug(self, **kwargs):
        pass

    def info(self, **kwargs):
        pass

    def warning(self, **kwargs):
        pass

    def error(self, **kwargs):
        pass

    def exception(self, exc_info=True, **kwargs):
        pass

    def critical(self, **kwargs):
        pass

    fatal = critical

    def log(self, level, **kwargs):
        pass


def print(*args):
    """
    Overwrites the builtin print function with `tqdm.tqdm.write`,
    so things are printed properly while tqdm is active.
    :param args:
    :return:
    """
    return tqdm.tqdm.write(' '.join(map(str, args)), file=sys.stdout)


BAR_FORMAT = '{desc} {percentage:3.0f}% {elapsed}<{remaining}, {rate_fmt}{postfix}'


class ProgressPrinter(tqdm.tqdm):
    def __call__(self, **kwargs):
        self.set_postfix(refresh=False, **kwargs)
        self.update()


def make_printer(bar_format=BAR_FORMAT, miniters=0,
                 mininterval=0.5, smoothing=0.1, file=sys.stdout, **kwargs):
    """
    Create a ProgressPrinter with appropriate parameters for training.
    See tqdm documentation for details on parameters.
    :return:

    """
    tqdm.tqdm.monitor_interval = 0
    p = ProgressPrinter(bar_format=bar_format, miniters=miniters,
                        mininterval=mininterval, smoothing=smoothing,
                        file=file,
                        **kwargs)
    return p
