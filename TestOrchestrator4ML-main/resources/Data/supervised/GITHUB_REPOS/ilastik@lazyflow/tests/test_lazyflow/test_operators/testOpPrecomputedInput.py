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
import numpy
from functools import partial
from lazyflow.graph import Graph, OperatorWrapper
from lazyflow.operators import OpArrayPiper, OpPixelOperator, OpPrecomputedInput


class TestOpPrecomputedInput(object):
    def setup_method(self, method):
        pass

    def testBasic(self):
        """
        Test the OpPrecomputedInput oeprator.
        """
        graph = Graph()
        dataShape = (1, 10, 10, 10, 1)
        data = numpy.indices(dataShape)
        precomputedData = data * 10

        computeCount = [0]

        def compute(x):
            computeCount[0] += 1
            return x * 10

        opSlow = OpPixelOperator(graph=graph)
        opSlow.Input.setValue(data)
        opSlow.Function.setValue(compute)

        opFast = OpArrayPiper(graph=graph)
        opFast.Input.setValue(precomputedData)

        op = OpPrecomputedInput(ignore_dirty_input=False, graph=graph)
        op.SlowInput.connect(opSlow.Output)

        # Should use slow input if no fast input is provided.
        computeCount[0] = 0
        result = op.Output[...].wait()
        assert (result == precomputedData).all()
        assert computeCount[0] == 1

        # Provide a fast input, which it should immediately start using.
        op.PrecomputedInput.connect(opFast.Output)

        # When input is still clean, it should use the fast input
        computeCount[0] = 0
        result = op.Output[...].wait()
        assert (result == precomputedData).all()
        assert computeCount[0] == 0

        # Subscribe to dirty output notifications
        dirtyRois = []

        def handleDirty(slot, roi):
            dirtyRois.append(roi)

        op.Output.notifyDirty(handleDirty)

        # Mark the slow input dirty.  Did the output see it?
        # (Output should see the dirty roi from the slow input even
        #  though it was previously connected to the fast input).
        opSlow.Input.setDirty(slice(None))
        assert len(dirtyRois) == 1

        # Op should switch to the slow input.
        result = op.Output[...].wait()
        assert (result == precomputedData).all()
        assert computeCount[0] == 1

        # Reset the operator.  Compute count should not change.
        computeCount[0] = 0
        op.Reset.setValue(True)
        op.Reset.setValue(False)
        result = op.Output[...].wait()
        assert (result == precomputedData).all()
        assert computeCount[0] == 0

    def testWrapped(self):
        """
        Make sure OpPrecomputedInput works even when wrapped in an OperatorWrapper.
        """
        graph = Graph()
        dataShape = (1, 10, 10, 10, 1)
        data = numpy.indices(dataShape).sum(0)

        precomputedData = [data * 10, data * 11, data * 12]

        computeCount = [0]

        def compute(m, x):
            computeCount[0] += 1
            return m * x

        opSlow = OperatorWrapper(OpPixelOperator, graph=graph)
        opSlow.Function.resize(3)
        opSlow.Function[0].setValue(partial(compute, 10))
        opSlow.Function[1].setValue(partial(compute, 11))
        opSlow.Function[2].setValue(partial(compute, 12))
        opSlow.Input.setValue(data)

        computeCount[0] = 0
        assert opSlow.Output.ready()
        assert opSlow.Output[0].ready()
        assert opSlow.Output[1].ready()

        opFast = OperatorWrapper(OpArrayPiper, graph=graph)
        opFast.Input.resize(3)
        opFast.Input[0].setValue(precomputedData[0])
        opFast.Input[1].setValue(precomputedData[1])
        opFast.Input[2].setValue(precomputedData[2])

        op = OperatorWrapper(OpPrecomputedInput, operator_kwargs=dict(ignore_dirty_input=False), graph=graph)
        op.PrecomputedInput.connect(opFast.Output)
        op.SlowInput.connect(opSlow.Output)

        assert opFast.Output.ready()
        assert opFast.Output[0].ready()
        assert opFast.Output[1].ready()
        assert opFast.Output[2].ready()

        assert len(op.SlowInput) == 3

        assert op.Output.ready()
        assert op.Output[0].ready()
        assert op.Output[1].ready()
        assert op.Output[2].ready()

        # When input is still clean, it should use the fast input
        result = op.Output[0][...].wait()
        assert (result == precomputedData[0]).all()
        result = op.Output[1][...].wait()
        assert (result == precomputedData[1]).all()
        result = op.Output[2][...].wait()
        assert (result == precomputedData[2]).all()
        assert computeCount[0] == 0

        # Mark input dirty.  Op should switch to the slow input.
        opSlow.Input[0].setDirty(slice(None))
        result = op.Output[0][...].wait()
        assert (result == precomputedData[0]).all()

        opSlow.Input[1].setDirty(slice(None))
        result = op.Output[1][...].wait()
        assert (result == precomputedData[1]).all()

        opSlow.Input[2].setDirty(slice(None))
        result = op.Output[2][...].wait()
        assert (result == precomputedData[2]).all()
        assert computeCount[0] == 3

        # Reset one of the operators.
        computeCount[0] = 0
        op.Reset[1].setValue(True)
        op.Reset[1].setValue(False)
        result = op.Output[0][...].wait()
        assert (result == precomputedData[0]).all()
        result = op.Output[1][...].wait()
        assert (result == precomputedData[1]).all()
        result = op.Output[2][...].wait()
        assert (result == precomputedData[2]).all()

        # Compute should still be needed for the 2 that didn't get reset
        assert computeCount[0] == 2


if __name__ == "__main__":
    import sys
    import nose

    sys.argv.append("--nocapture")  # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture")  # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret:
        sys.exit(1)
