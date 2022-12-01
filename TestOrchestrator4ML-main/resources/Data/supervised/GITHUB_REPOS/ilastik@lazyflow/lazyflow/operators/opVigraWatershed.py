from builtins import zip

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
from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy
import vigra
import logging
import threading
import warnings

logger = logging.getLogger(__name__)


class OpVigraWatershed(Operator):
    """
    Operator wrapper for vigra's default watershed function.
    """

    name = "OpVigraWatershed"
    category = "Vigra"

    InputImage = InputSlot()
    PaddingWidth = (
        InputSlot()
    )  # Specifies the extra pixels around the border of the image to use when computing the watershed.
    # (Region is clipped to the size of the input image.)

    SeedImage = InputSlot(optional=True)

    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpVigraWatershed, self).__init__(*args, **kwargs)

        # Keep a dict of roi : max label
        self._maxLabels = {}
        self._lock = threading.Lock()

    @property
    def maxLabels(self):
        return self._maxLabels

    def clearMaxLabels(self):
        with self._lock:
            self._maxLabels = {}

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.InputImage.meta)
        self.Output.meta.dtype = numpy.uint32

        # warnings.warn("FIXME: How can this drange be right?")
        # self.Output.meta.drange = (0,255)

        if self.SeedImage.ready():
            assert numpy.issubdtype(self.SeedImage.meta.dtype, numpy.uint32)
            assert self.SeedImage.meta.shape == self.InputImage.meta.shape, "{} != {}".format(
                self.SeedImage.meta.shape, self.InputImage.meta.shape
            )

    def getSlicings(self, roi):
        """
        Pad the given roi to obtain a new slicing to use for obtaining input data.
        Return the padded slicing and the slicing that returns the original roi within the padded data.
        """
        tags = self.InputImage.meta.axistags
        pairs = list(zip([tag.key for tag in tags], list(zip(roi.start, roi.stop))))
        slices = [(k, slice(start, stop)) for (k, (start, stop)) in pairs]

        # Compute the watershed over a larger area than requested (padded area)
        padding = self.PaddingWidth.value
        paddedSlices = []  # The requested slicing + padding
        outputSlices = []  # The slicing to get the requested slicing from the padded data
        for i, (key, s) in enumerate(slices):
            p = s
            if key in "xyz":
                p_start = max(s.start - padding, 0)
                p_stop = min(s.stop + padding, self.InputImage.meta.shape[i])
                p = slice(p_start, p_stop)

            paddedSlices += [p]
            o = slice(s.start - p.start, s.stop - p.start)
            outputSlices += [o]

        return paddedSlices, outputSlices

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Output

        # Every request is computed on-the-fly.
        # (No caching)
        paddedSlices, outputSlices = self.getSlicings(roi)

        # Get input data
        inputRegion = self.InputImage[paddedSlices].wait()

        # Makes sure vigra will understand this type
        if inputRegion.dtype != numpy.uint8 and inputRegion.dtype != numpy.float32:
            inputRegion = inputRegion.astype("float32")

        # Convert to vigra array
        inputRegion = inputRegion.view(vigra.VigraArray)
        inputRegion.axistags = self.InputImage.meta.axistags

        # Reduce to 3-D (keep order of xyz axes)
        tags = self.InputImage.meta.axistags
        axes3d = "".join([tag.key for tag in tags if tag.key in "xyz"])
        inputRegion = inputRegion.withAxes(*axes3d)
        logger.debug("inputRegion 3D shape:{}".format(inputRegion.shape))

        logger.debug("roi={}".format(roi))
        logger.debug("paddedSlices={}".format(paddedSlices))
        logger.debug("outputSlices={}".format(outputSlices))

        # If we know the range of the data, then convert to uint8
        # so we can automatically benefit from vigra's "turbo" mode
        if self.InputImage.meta.drange is not None:
            drange = self.InputImage.meta.drange
            inputRegion = numpy.asarray(inputRegion, dtype=numpy.float32)
            inputRegion = vigra.taggedView(inputRegion, axes3d)
            inputRegion -= drange[0]
            inputRegion /= drange[1] - drange[0]
            inputRegion *= 255.0
            inputRegion = inputRegion.astype(numpy.uint8)

        # This is where the magic happens
        if self.SeedImage.ready():
            seedImage = self.SeedImage[paddedSlices].wait()
            seedImage = seedImage.view(vigra.VigraArray)
            seedImage.axistags = tags
            seedImage = seedImage.withAxes(*axes3d)
            logger.debug("Input shape = {}, seed shape = {}".format(inputRegion.shape, seedImage.shape))
            logger.debug("Input axes = {}, seed axes = {}".format(inputRegion.axistags, seedImage.axistags))
            watershed, maxLabel = vigra.analysis.watersheds(inputRegion, seeds=seedImage)
        else:
            watershed, maxLabel = vigra.analysis.watersheds(inputRegion)
        logger.debug("Finished Watershed")

        logger.debug("watershed 3D output shape={}".format(watershed.shape))
        logger.debug("maxLabel={}".format(maxLabel))

        # Promote back to 5-D
        watershed = vigra.taggedView(watershed, axes3d)
        watershed = watershed.withAxes(*[tag.key for tag in tags])
        logger.debug("watershed 5D shape: {}".format(watershed.shape))
        logger.debug("watershed axistags: {}".format(watershed.axistags))

        with self._lock:
            start = tuple(s.start for s in paddedSlices)
            stop = tuple(s.stop for s in paddedSlices)
            self._maxLabels[(start, stop)] = maxLabel

        # print numpy.sort(vigra.analysis.unique(watershed[outputSlices])).shape
        # Return only the region the user requested
        result[:] = watershed[outputSlices].view(numpy.ndarray).reshape(result.shape)
        return result

    def propagateDirty(self, inputSlot, subindex, roi):
        if not self.configured():
            self.Output.setDirty(slice(None))
        elif inputSlot.name == "InputImage" or inputSlot.name == "SeedImage":
            paddedSlicing, outputSlicing = self.getSlicings(roi)
            self.Output.setDirty(paddedSlicing)
        elif inputSlot.name == "PaddingWidth":
            self.Output.setDirty(slice(None))
        else:
            assert False, "Unknown input slot."
