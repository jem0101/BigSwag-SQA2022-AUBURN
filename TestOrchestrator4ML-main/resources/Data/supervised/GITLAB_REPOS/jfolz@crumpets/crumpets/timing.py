from __future__ import print_function, division, absolute_import

import time
from datetime import datetime
from datetime import timedelta


def nicetime(seconds):
    """
    formats given time in form of seconds to a nice string representation.

    :return: string representation of time
    """

    hours = seconds // 3600
    minutes = seconds % 3600 // 60
    seconds = seconds % 60
    if hours:
        return '%d:%02d:%02d hours' % (hours, minutes, seconds)
    elif minutes:
        return '%d:%02d minutes' % (minutes, seconds)
    else:
        return '%d seconds' % seconds


class RemainingTimer(object):
    """
    Estimates remaining time of some operation.
    Each time it is called, an internal counter is increased by 1.
    The measured operation is assumed to be done once this counter reaches goal.
    Out of the frequency of the calls one can compute an estimated speed and thus remaining time.

    :param goal: the number of calls until the operation is assumed to be done.
    """

    def __init__(self, goal):
        self.start = time.time()
        self.i = 0
        self.goal = goal

    def _rem(self):
        now = time.time()
        n = self.goal
        self.i = self.i
        self.i += 1
        delta = now - self.start
        speed = delta / self.i
        return speed * (n - self.i)

    def __call__(self):
        return nicetime(self._rem())


class ETATimer(RemainingTimer):
    """
    Simple modification of the RemainingTimer.
    When called, instead of returning remainin time, return ETA (estimated time of arrival).
    """
    def __call__(self):
        now = datetime.now()
        delta = timedelta(seconds=self._rem())
        done = now + delta
        return done.replace(microsecond=0)
