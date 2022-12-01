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
import os
import copy
import tempfile
import h5py
import vigra
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.utility.io_util.RESTfulVolume import RESTfulVolume
import logging

logger = logging.getLogger(__name__)


class OpRESTfulVolumeReader(Operator):
    """
    An operator to retrieve hdf5 volumes from a remote server that provides a RESTful interface.
    The operator requires a LOCAL json config file that describes the remote dataset and interface.
    """

    name = "OpRESTfulVolumeReader"

    DescriptionFilePath = InputSlot(stype="filestring")
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpRESTfulVolumeReader, self).__init__(*args, **kwargs)
        self._axes = None
        self._volumeObject = None

    def setupOutputs(self):
        # Create a RESTfulVolume object to read the description file and do the downloads.
        self._volumeObject = RESTfulVolume(self.DescriptionFilePath.value)

        self._axes = self._volumeObject.description.axes
        outputShape = tuple(self._volumeObject.description.shape)

        # If the dataset has no channel axis, add one.
        if "c" not in self._axes:
            outputShape += (1,)
            self._axes += "c"

        self.Output.meta.shape = outputShape
        self.Output.meta.dtype = self._volumeObject.description.dtype
        self.Output.meta.axistags = vigra.defaultAxistags(str(self._axes))

    def execute(self, slot, subindex, roi, result):
        roi = copy.copy(roi)

        # If we are artificially adding a channel index, remove it from the roi for the download.
        if len(self.Output.meta.shape) > len(self._volumeObject.description.shape):
            roi.start.pop(self.Output.meta.axistags.index("c"))
            roi.stop.pop(self.Output.meta.axistags.index("c"))

        # Write the data from the url out to disk (in a temporary file)
        hdf5FilePath = os.path.join(tempfile.mkdtemp(), "cube.h5")
        hdf5DatasetPath = hdf5FilePath + self._volumeObject.description.hdf5_dataset
        self._volumeObject.downloadSubVolume((roi.start, roi.stop), hdf5DatasetPath)

        # Open the file we just created using h5py
        with h5py.File(hdf5FilePath, "r") as hdf5File:
            dataset = hdf5File[self._volumeObject.description.hdf5_dataset]
            if len(result.shape) > len(dataset.shape):
                # We appended a channel axis to Output, but the dataset doesn't have that.
                result[..., 0] = dataset[...]
            else:
                result[...] = dataset[...]
        return result

    def propagateDirty(self, slot, subindex, roi):
        assert slot == self.DescriptionFilePath, "Unknown input slot."
        self.Output.setDirty(slice(None))
