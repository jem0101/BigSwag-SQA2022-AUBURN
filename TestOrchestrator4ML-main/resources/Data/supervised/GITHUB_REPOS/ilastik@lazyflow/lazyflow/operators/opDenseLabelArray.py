from builtins import map

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
import numpy
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiToSlice


class OpDenseLabelArray(Operator):
    """
    The simplest, most naive implementation of a labeling operator.

    - Does not track max label value correctly
    - Does not ensure consecutive labeling (i.e. If you delete a label, the other labels are not 'shifted down'.
    """

    MetaInput = InputSlot()
    LabelSinkInput = InputSlot(optional=True)
    EraserLabelValue = InputSlot(value=255)  # Value slot.  Specifies the magic 'eraser' label.

    DeleteLabel = InputSlot(value=-1)  # If > 0, remove that label from the array.

    Output = OutputSlot()
    MaxLabelValue = OutputSlot()  # Hard-coded for now
    NonzeroBlocks = OutputSlot()  # list of slicings

    def __init__(self, *args, **kwargs):
        super(OpDenseLabelArray, self).__init__(*args, **kwargs)
        self._cache = None

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.MetaInput.meta)
        self.Output.meta.dtype = numpy.uint8

        assert self.MetaInput.meta.getAxisKeys()[-1] == "c", "This operator assumes that the last axis must be channel."
        self.Output.meta.shape = self.MetaInput.meta.shape[:-1] + (1,)

        self.MaxLabelValue.meta.dtype = numpy.uint8
        self.MaxLabelValue.meta.shape = (1,)

        self.NonzeroBlocks.meta.dtype = object
        self.NonzeroBlocks.meta.shape = (1,)

        assert self.EraserLabelValue.value != 0, "Eraser label value must be non-zero."

        if self._cache is None or self._cache.shape != self.Output.meta.shape:
            self._cache = numpy.zeros(self.Output.meta.shape, dtype=self.Output.meta.dtype)

        delete_label_value = self.DeleteLabel.value
        if self.DeleteLabel.value != -1:
            self._cache[self._cache == delete_label_value] = 0

    def execute(self, slot, subindex, roi, destination):
        if slot == self.Output:
            destination[:] = self._cache[roiToSlice(roi.start, roi.stop)]
        elif slot == self.MaxLabelValue:
            # FIXME: Don't hard-code this
            destination[0] = 2
        elif slot == self.NonzeroBlocks:
            # Only one block, the bounding box for all non-zero values.
            # This is efficient if the labels are very close to eachother,
            #  but slow if the labels are far apart.
            nonzero_coords = numpy.nonzero(self._cache)
            if len(nonzero_coords) > 0 and len(nonzero_coords[0]) > 0:
                bounding_box_start = numpy.array(list(map(numpy.min, nonzero_coords)))
                bounding_box_stop = 1 + numpy.array(list(map(numpy.max, nonzero_coords)))
                destination[0] = [roiToSlice(bounding_box_start, bounding_box_stop)]
            else:
                destination[0] = []
        else:
            assert False, "Unknown output slot: {}".format(slot.name)
        return destination

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.LabelSinkInput:
            self.Output.setDirty(*roi)
        if slot == self.DeleteLabel and self.DeleteLabel.value != -1:
            self.Output.setDirty()
            self.NonzeroBlocks.setDirty()

    def setInSlot(self, slot, subindex, roi, value):
        if slot == self.LabelSinkInput:
            # Extract the data to modify
            orig_block = self._cache[roiToSlice(roi.start, roi.stop)]

            # Reset the pixels we need to change
            orig_block[value.nonzero()] = 0

            # Update
            orig_block |= value

            # Replace 'eraser' values with zeros.
            cleaned_block = numpy.where(orig_block == self.EraserLabelValue.value, 0, orig_block[:])

            # Set in the cache.
            self._cache[roiToSlice(roi.start, roi.stop)] = cleaned_block
            self.Output.setDirty(roi.start, roi.stop)
