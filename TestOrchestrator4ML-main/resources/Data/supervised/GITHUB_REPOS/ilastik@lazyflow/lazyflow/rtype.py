from future import standard_library

standard_library.install_aliases()
from builtins import map
from builtins import zip

from builtins import range
from builtins import object

###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
# 		   http://ilastik.org/license/
###############################################################################
import numpy, copy
import pickle as pickle
import collections
import sys

from lazyflow.roi import TinyVector, sliceToRoi, roiToSlice, roiFromShape
from lazyflow.utility import slicingtools

import logging
from future.utils import with_metaclass

logger = logging.getLogger(__name__)


class RoiMeta(type):
    """
    Roi metaclass.  This mechanism automatically registers all roi
    subclasses for string serialization via Roi._registerSubclass.
    """

    def __new__(cls, name, bases, classDict):
        cls = super(RoiMeta, cls).__new__(cls, name, bases, classDict)
        # Don't register the Roi baseclass itself.
        if bases[0] != object:
            Roi._registerSubclass(cls)
        return cls


class Roi(with_metaclass(RoiMeta, object)):
    def __init__(self, slot):
        self.slot = slot
        pass

    pass

    all_subclasses = set()

    @classmethod
    def _registerSubclass(cls, roiType):
        Roi.all_subclasses.add(roiType)

    @staticmethod
    def _toString(roi):
        """
        Convert roi into a string that can be converted back into a roi via _fromString().
        The default implementation uses pickle.
        Subclasses may override this and _fromString for more human-friendly string representations.
        """
        return pickle.dumps(roi, 0)

    @staticmethod
    def _fromString(s):
        """
        Convert string 's' into a roi.
        The default implementation uses pickle.
        Subclasses may override this and _toString for more human-friendly string representations.
        """
        return pickle.loads(s)

    @staticmethod
    def dumps(roi):
        return roi.__class__.__name__ + ":" + roi.__class__._toString(roi)

    @staticmethod
    def loads(s):
        for cls in Roi.all_subclasses:
            if s.startswith(cls.__name__):
                return cls._fromString(s[len(cls.__name__) + 1 :])
        assert False, "Class name within '{}' does not refer to any Roi subclasses.".format(s)


class Everything(Roi):
    """Fallback Roi for Slots that can't operate on subsets of their input data."""

    def __init__(self, slot, *args, **kwargs):
        self.slot = slot


class List(Roi):
    def __init__(self, slot, iterable=(), pslice=None):
        super(List, self).__init__(slot)
        self._l = list(iterable)
        if pslice is not None:
            logger.debug("pslice not none, but we are in a list! {}".format(pslice))
            self._l = [pslice]

    def __iter__(self):
        return iter(self._l)

    def __len__(self):
        return len(self._l)

    def __str__(self):
        return str(self._l)


class SubRegion(Roi):
    def __init__(self, slot, start=None, stop=None, pslice=None):
        super(SubRegion, self).__init__(slot)
        shape = None
        if slot is not None:
            shape = slot.meta.shape
        if pslice != None or start is not None and stop is None and pslice is None:
            if pslice is None:
                pslice = start
            if shape is None:
                # Okay to use a shapeless slot if the key is bounded
                # AND if the key has the correct length
                assert slicingtools.is_bounded(pslice)
                # Supply a dummy shape
                shape = [sl.stop for sl in pslice]
            self.start, self.stop = sliceToRoi(pslice, shape)
        elif start is None and pslice is None:
            assert shape is not None, "Can't create a default subregion without a slot and a shape."
            self.start, self.stop = roiFromShape(shape)
        else:
            self.start = TinyVector(start)
            self.stop = TinyVector(stop)
        self.dim = len(self.start)

        for start, stop in zip(self.start, self.stop):
            assert isinstance(start, (int, numpy.integer)), "Roi contains non-integers: {}".format(self)
            assert isinstance(start, (int, numpy.integer)), "Roi contains non-integers: {}".format(self)

    # FIXME: This assertion is good at finding bugs, but it is currently triggered by
    #        the DataExport applet when the output axis order is changed.
    #
    #         if self.slot is not None self.slot.meta.shape is not None:
    #             assert all(self.stop <= self.slot.meta.shape), \
    #                 "Roi is out of bounds. roi={}, {}.{}.meta.shape={}"\
    #                 .format((self.start, self.stop), slot.getRealOperator().name, slot.name, self.slot.meta.shape)

    def __setstate__(self, state):
        """
        Support copy.copy()
        """
        self.slot = state["slot"]
        self.start = TinyVector(state["start"])
        self.stop = TinyVector(state["stop"])
        self.dim = len(state["start"])

    def __str__(self):
        return "".join(("Subregion: start '", str(self.start), "' stop '", str(self.stop), "'"))

    def pprint(self):
        """pretty-print this object"""
        ret = ""
        for a, b in zip(self.start, self.stop):
            ret += "%d-%d " % (a, b)
        return ret

    @staticmethod
    def _toString(roi):
        assert isinstance(roi, SubRegion)
        assert roi.slot is None, "Can't stringify SubRegions with no slot"
        return "SubRegion(None, {}, {})".format(roi.start, roi.stop)

    @staticmethod
    def _fromString(s):
        return eval(s)

    def setInputShape(self, inputShape):
        assert type(inputShape) == tuple
        self.inputShape = inputShape

    def copy(self):
        return copy.copy(self)

    def popDim(self, dim):
        """
        remove the i'th dimension from the SubRegion
        works inplace !
        """
        if dim is not None:
            self.start.pop(dim)
            self.stop.pop(dim)
        return self

    def setDim(self, dim, start, stop):
        """
        change the subarray at dim, to begin at start
        and to end at stop
        """
        self.start[dim] = start
        self.stop[dim] = stop
        return self

    def insertDim(self, dim, start, stop):
        """
        insert a new dimension before dim.
        set start to start, stop to stop
        and the axistags to at
        """
        self.start = self.start.insert(dim, start)
        self.stop = self.stop.insert(dim, stop)
        return self

    def expandByShape(self, shape, cIndex, tIndex):
        """
        extend a roi by a given in shape
        """
        # TODO: Warn if bounds are exceeded
        cStart = self.start[cIndex]
        cStop = self.stop[cIndex]
        if tIndex is not None:
            tStart = self.start[tIndex]
            tStop = self.stop[tIndex]
        if isinstance(shape, collections.Iterable):
            # add a dummy number for the channel dimension
            shape = shape + (1,)
        else:
            tmp = shape
            shape = numpy.zeros(self.dim).astype(int)
            shape[:] = tmp

        tmpStart = [int(x - s) for x, s in zip(self.start, shape)]
        tmpStop = [int(x + s) for x, s in zip(self.stop, shape)]
        start = [int(max(t, i)) for t, i in zip(tmpStart, numpy.zeros_like(self.inputShape))]
        stop = [int(min(t, i)) for t, i in zip(tmpStop, self.inputShape)]
        start[cIndex] = cStart
        stop[cIndex] = cStop
        if tIndex is not None:
            start[tIndex] = tStart
            stop[tIndex] = tStop
        self.start = TinyVector(start)
        self.stop = TinyVector(stop)
        return self

    def adjustRoi(self, halo, cIndex=None):
        if type(halo) != list:
            halo = [halo] * len(self.start)
        notAtStartEgde = list(map(lambda x, y: True if x < y else False, halo, self.start))
        for i in range(len(notAtStartEgde)):
            if notAtStartEgde[i]:
                self.stop[i] = int(self.stop[i] - self.start[i] + halo[i])
                self.start[i] = int(halo[i])
        return self

    def adjustChannel(self, cPerC, cIndex, channelRes):
        if cPerC != 1 and channelRes == 1:
            start = [self.start[i] // cPerC if i == cIndex else self.start[i] for i in range(len(self.start))]
            stop = [self.stop[i] // cPerC + 1 if i == cIndex else self.stop[i] for i in range(len(self.stop))]
            self.start = TinyVector(start)
            self.stop = TinyVector(stop)
        elif channelRes > 1:
            start = [0 if i == cIndex else self.start[i] for i in range(len(self.start))]
            stop = [channelRes if i == cIndex else self.stop[i] for i in range(len(self.stop))]
            self.start = TinyVector(start)
            self.stop = TinyVector(stop)
        return self

    def toSlice(self):
        return roiToSlice(self.start, self.stop)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.dim != other.dim:
            return False
        starts_equal = numpy.all(self.start == other.start)
        stops_equal = numpy.all(self.stop == other.stop)
        return starts_equal and stops_equal
