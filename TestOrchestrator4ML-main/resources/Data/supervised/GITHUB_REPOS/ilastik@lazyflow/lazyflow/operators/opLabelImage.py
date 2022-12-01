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

import warnings

# Third-party
import numpy
import vigra

# Lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot, OperatorWrapper
from lazyflow.operators import OpMultiArraySlicer2, OpMultiArrayStacker
from lazyflow.operators.opVigraLabelVolume import _OpVigraLabelVolume as OpVigraLabelVolume


class _OpLabelImage(Operator):
    """
    Produces labeled 5D volumes.  If multiple time slices and/or channels are present,
    each time/channel combo is treated as a separate volume for labeling,
    which are then stacked at the output.
    """

    Input = InputSlot()
    BackgroundLabels = InputSlot(optional=True)  # Must be a list: one for each channel of the volume.

    Output = OutputSlot()

    # Schematic:
    #
    # BackgroundLabels -> opBgTimeSlicer -> opBgChannelSlicer ----
    #                                                             \
    # Input ------------> opTimeSlicer ---> opChannelSlicer -----> opLabelers -> opChannelStacker -> opTimeStacker -> Output

    def __init__(self, *args, **kwargs):
        """
        Set up the internal pipeline.
        Since each labeling operator can only handle a single time and channel,
        we split the volume along time and channel axes to produce N 3D volumes, where N=T*C.
        The volumes are combined again into a 5D volume on the output using stackers.

        See ascii schematic in comments above for an overview.
        """
        super(_OpLabelImage, self).__init__(*args, **kwargs)

        self.opTimeSlicer = OpMultiArraySlicer2(parent=self)
        self.opTimeSlicer.AxisFlag.setValue("t")
        self.opTimeSlicer.Input.connect(self.Input)
        assert self.opTimeSlicer.Slices.level == 1

        self.opChannelSlicer = OperatorWrapper(OpMultiArraySlicer2, parent=self)
        self.opChannelSlicer.AxisFlag.setValue("c")
        self.opChannelSlicer.Input.connect(self.opTimeSlicer.Slices)
        assert self.opChannelSlicer.Slices.level == 2

        class OpWrappedVigraLabelVolume(Operator):
            """
            This quick hack is necessary because there's not currently a way to wrap an OperatorWrapper.
            We need to double-wrap OpVigraLabelVolume, so we need this operator to provide the first level of wrapping.
            """

            Input = InputSlot(level=1)
            BackgroundValue = InputSlot(optional=True, level=1)

            Output = OutputSlot(level=1)

            def __init__(self, *args, **kwargs):
                super(OpWrappedVigraLabelVolume, self).__init__(*args, **kwargs)
                self._innerOperator = OperatorWrapper(OpVigraLabelVolume, parent=self)
                self._innerOperator.Input.connect(self.Input)
                self._innerOperator.BackgroundValue.connect(self.BackgroundValue)
                self.Output.connect(self._innerOperator.Output)

            def execute(self, slot, subindex, roi, destination):
                assert False, "Shouldn't get here."

            def propagateDirty(self, slot, subindex, roi):
                pass  # Nothing to do...

        # Wrap OpVigraLabelVolume TWICE.
        self.opLabelers = OperatorWrapper(OpWrappedVigraLabelVolume, parent=self)
        assert self.opLabelers.Input.level == 2
        self.opLabelers.Input.connect(self.opChannelSlicer.Slices)

        # The background labels will be converted to a VigraArray with axistags 'tc' so they can
        # be distributed to the labeling operators via slicers in the same manner as the input data.
        # Here, we set up the slicers that will distribute the background labels to the appropriate labelers.
        self.opBgTimeSlicer = OpMultiArraySlicer2(parent=self)
        self.opBgTimeSlicer.AxisFlag.setValue("t")
        assert self.opBgTimeSlicer.Slices.level == 1

        self.opBgChannelSlicer = OperatorWrapper(OpMultiArraySlicer2, parent=self)
        self.opBgChannelSlicer.AxisFlag.setValue("c")
        self.opBgChannelSlicer.Input.connect(self.opBgTimeSlicer.Slices)
        assert self.opBgChannelSlicer.Slices.level == 2

        assert self.opLabelers.BackgroundValue.level == 2
        self.opLabelers.BackgroundValue.connect(self.opBgChannelSlicer.Slices)

        self.opChannelStacker = OperatorWrapper(OpMultiArrayStacker, parent=self)
        self.opChannelStacker.AxisFlag.setValue("c")

        assert self.opLabelers.Output.level == 2
        assert self.opChannelStacker.Images.level == 2
        self.opChannelStacker.Images.connect(self.opLabelers.Output)

        self.opTimeStacker = OpMultiArrayStacker(parent=self)
        self.opTimeStacker.AxisFlag.setValue("t")

        assert self.opChannelStacker.Output.level == 1
        assert self.opTimeStacker.Images.level == 1
        self.opTimeStacker.Images.connect(self.opChannelStacker.Output)

        # Connect our outputs
        self.Output.connect(self.opTimeStacker.Output)

    def setupOutputs(self):
        assert set(self.Input.meta.getTaggedShape().keys()) == set(
            "txyzc"
        ), "OpLabelImage requires all txyzc axes to be present in the input."

        # These slots couldn't be configured in __init__ because Input wasn't connected yet.
        self.opChannelStacker.AxisIndex.setValue(self.Input.meta.axistags.index("c"))
        self.opTimeStacker.AxisIndex.setValue(self.Input.meta.axistags.index("t"))

        taggedShape = self.Input.meta.getTaggedShape()
        if self.BackgroundLabels.ready():
            # Turn this list into an array with axistags='tc' that can be sliced by time and channel,
            #  just like the input data
            bgLabelList = self.BackgroundLabels.value
            assert (
                len(bgLabelList) == taggedShape["c"]
            ), "If background labels are provided, there must be one for each input channel"
            bgLabelVolume = numpy.ndarray(shape=(taggedShape["t"], taggedShape["c"]), dtype=numpy.uint32)

            # Duplicate the bg label list for all time slices
            bgLabelVolume[...] = bgLabelList
            bgLabelVolume = bgLabelVolume.view(vigra.VigraArray)
            bgLabelVolume.axistags = vigra.defaultAxistags("tc")
            self.opBgTimeSlicer.Input.setValue(bgLabelVolume)
        else:
            self.opBgTimeSlicer.Input.disconnect()

    def execute(self, slot, subindex, roi, destination):
        assert False, "Shouldn't get here."

    def propagateDirty(self, slot, subindex, roi):
        pass  # Nothing to do...


class OpLabelImage(_OpLabelImage):
    def __init__(self, *args, **kwargs):
        warnings.warn("OpLabelImage is deprecated, use OpLabelVolume instead", DeprecationWarning)
        super(OpLabelImage, self).__init__(*args, **kwargs)
