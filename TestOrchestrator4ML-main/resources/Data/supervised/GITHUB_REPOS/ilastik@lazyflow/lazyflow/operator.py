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
import collections
import functools
import logging
import threading
import sys
import inspect

from abc import ABCMeta
from contextlib import contextmanager
from traceback import walk_tb, FrameSummary, format_list

# lazyflow
from lazyflow.slot import InputSlot, OutputSlot, Slot


class InputDict(collections.OrderedDict):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def __setitem__(self, key, value):
        assert isinstance(
            value, InputSlot
        ), "ERROR: all elements of .inputs must be of type InputSlot." " You provided {}!".format(value)
        return super(InputDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key in self:
            return super(InputDict, self).__getitem__(key)
        elif hasattr(self.operator, key):
            return getattr(self.operator, key)
        else:
            raise Exception(
                "Operator {} (class: {}) has no input slot named '{}'."
                " Available input slots are: {}".format(
                    self.operator.name, self.operator.__class__, key, list(self.keys())
                )
            )


class OutputDict(collections.OrderedDict):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def __setitem__(self, key, value):
        assert isinstance(
            value, OutputSlot
        ), "ERROR: all elements of .outputs must be of type" " OutputSlot. You provided {}!".format(value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        elif hasattr(self.operator, key):
            return getattr(self.operator, key)
        else:
            raise Exception(
                "Operator {} (class: {}) has no output slot named '{}'."
                " Available output slots are: {}".format(
                    self.operator.name, self.operator.__class__, key, list(self.keys())
                )
            )


class OperatorMetaClass(ABCMeta):
    def __new__(cls, name, bases, classDict):
        cls = super().__new__(cls, name, bases, classDict)

        # this allows for definition of input-/ output-slots the following way:
        #    inputSlots = [InputSlot("MySlot"), InputSlot("MySlot2")]
        # This was the original type of slot definition but should not be invoked
        # by users for their workflow.
        # For OperatorSubview, however, this is essential
        setattr(cls, "inputSlots", list(cls.inputSlots))
        setattr(cls, "outputSlots", list(cls.outputSlots))

        # Slots in operators should be defined in the "fancy" syntax (see below).
        # Support fancy syntax.
        # If the user typed this in his class definition:
        #    MySlot = InputSlot()
        #    MySlot2 = InputSlot()

        for k, v in list(cls.__dict__.items()):
            if isinstance(v, InputSlot):
                v.name = k
                cls.inputSlots.append(v)

            if isinstance(v, OutputSlot):
                v.name = k
                cls.outputSlots.append(v)
        return cls

    def __call__(cls, *args, **kwargs):
        # type.__call__ calls instance.__init__ internally
        try:
            instance = ABCMeta.__call__(cls, *args, **kwargs)
        except Exception as e:
            # FIXME: What is the point of this long exception message?
            # Why can't we just let the exception propagate up the stack?
            # Is it because of some weird interaction between this metaclass and the Qt event loop?
            # ....probably
            err = "Could not create instance of '{}'\n".format(cls)
            err += "args   = {}\n".format(args)
            err += "kwargs = {}\n".format(kwargs)
            err += "The exception was:\n"
            err += str(e)
            err += "\nTraceback:\n"
            import traceback
            import sys
            import io

            if sys.version_info.major == 2:
                s = io.BytesIO()
            else:
                s = io.StringIO()
            traceback.print_exc(file=s)
            err += s.getvalue()
            raise RuntimeError(err)
        instance._after_init()
        return instance


class Operator(metaclass=OperatorMetaClass):
    """The base class for all Operators.

    Operators consist of a class inheriting from this class
    and need to specify their inputs and outputs via
    thei inputSlot and outputSlot class properties.

    Each instance of an operator obtains individual
    copies of the inputSlots and outputSlots, which are
    available in the self.inputs and self.outputs instance
    properties.

    these instance properties can be used to connect
    the inputs and outputs of different operators.

    Example:
        operator1.inputs["InputA"].connect(operator2.outputs["OutputC"])


    Different examples for simple operators are provided
    in an example directory. plese read through the
    examples to learn how to implement your own operators...

    Each operator instance is associated with a lazyflow.graph.Graph object to track
    dependencies between operators. An operator instance either
    inherits the graph from its parent or---if it has no parent---is
    assigned to a graph instance directly. The dependency tracking is
    mainly used for debugging purposes and to diagnose a network of operators.

    """

    loggerName = __name__ + ".Operator"
    logger = logging.getLogger(loggerName)
    traceLogger = logging.getLogger("TRACE." + loggerName)

    # definition of inputs slots
    inputSlots = []

    # definition of output slots -> operators instances
    outputSlots = []
    name = "Operator (base class)"
    description = ""
    category = "lazyflow"

    @property
    def transaction(self):
        """
        Create transaction for this operation deferring setupOutputs call
        until transaction is finished
        :returns: Transaction context manager
        """
        return self.graph.transaction

    def __new__(cls, *args, **kwargs):
        ##
        # before __init__
        ##
        obj = object.__new__(cls)
        obj.inputs = InputDict(obj)
        obj.outputs = OutputDict(obj)
        return obj

    def __init__(self, parent=None, graph=None):
        """
        Either parent or graph have to be given. If both are given
        parent.graph has to be identical with graph.

        :param parent: the parent operator; if None the instance is a
        root operator
        :param graph: a Graph instance

        """
        if not (parent is None or isinstance(parent, Operator)):
            raise Exception(
                "parent of operator name='{}' must be an operator,"
                " not {} of type {}".format(self.name, parent, type(parent))
            )
        if parent and graph and parent.graph is not graph:
            raise Exception("graph of parent and graph of operator name='%s' have to be the same" % self.name)
        if graph is None:
            if parent is None:
                raise Exception(
                    "Operator.__init__() [self.name='{}']:" " parent and graph can't be both None".format(self.name)
                )
            graph = parent.graph

        self._cleaningUp = False
        self.graph = graph
        self._children = collections.OrderedDict()
        self._parent = None
        if parent is not None:
            parent._add_child(self)

        self._initialized = False

        self._condition = threading.Condition()
        self._executionCount = 0
        self._settingUp = False

        self._instantiate_slots()

        # We normally assert if an operator's upstream partners are
        # yanked away. If operator is marked as "externally_managed",
        # then we'll avoid the assert. In that case, it's assumed that
        # you know what you're doing, and you weren't planning to use
        # that operator, anyway.
        self.externally_managed = False

        self._debug_text = None
        self._setup_count = 0

    @property
    def children(self):
        return list(self._children.keys())

    def _add_child(self, child):
        # We're just using an OrderedDict for O(1) lookup with
        # in-order iteration but we don't actually store any values
        assert child.parent is None
        self._children[child] = None
        child._parent = self

    # continue initialization, when user overrides __init__
    def _after_init(self):
        # provide simple default name for lazy users
        if self.name == Operator.name:
            self.name = type(self).__name__
        assert self.graph is not None, (
            "Operator {}: self.graph is None, the parent ({})"
            " given to the operator must have a valid .graph attribute!".format(self, self._parent)
        )
        # check for slot uniqueness
        temp = {}
        for i in self.inputSlots:
            if i.name in temp:
                raise Exception(
                    "ERROR: Operator {} has multiple slots with name {},"
                    " please make sure that all input and output slot"
                    " names are unique".format(self.name, i.name)
                )
            temp[i.name] = True

        for i in self.outputSlots:
            if i.name in temp:
                raise Exception(
                    "ERROR: Operator {} has multiple slots with name {},"
                    " please make sure that all input and output slot"
                    " names are unique".format(self.name, i.name)
                )
            temp[i.name] = True

        self._instantiate_slots()

        self._setDefaultInputValues()

        for islot in list(self.inputs.values()):
            islot.notifyUnready(self.handleInputBecameUnready)

        self._initialized = True
        self._setupOutputs()

    def _instantiate_slots(self):
        # replicate input slot connections
        # defined for the operator for the instance
        for i in sorted(self.inputSlots, key=lambda s: s._global_slot_id):
            if i.name not in self.inputs:
                ii = i._getInstance(self)
                ii.connect(i.upstream_slot)
                self.inputs[i.name] = ii

        for k, v in list(self.inputs.items()):
            self.__dict__[k] = v

        # relicate output slots
        # defined for the operator for the instance
        for o in sorted(self.outputSlots, key=lambda s: s._global_slot_id):
            if o.name not in self.outputs:
                oo = o._getInstance(self)
                self.outputs[o.name] = oo

        for k, v in list(self.outputs.items()):
            self.__dict__[k] = v

    @property
    def parent(self):
        return self._parent

    def __setattr__(self, name, value):
        """This method safeguards that operators do not overwrite slot
        names with custom instance attributes.

        """
        if "inputs" in self.__dict__ and "outputs" in self.__dict__:
            if name in self.inputs or name in self.outputs:
                assert isinstance(value, Slot), (
                    "ERROR: trying to set attribute {} of operator {}"
                    " to value {}, which is not of type Slot !".format(name, self, value)
                )
        object.__setattr__(self, name, value)

    def configured(self):
        """Returns True if all input slots that are non-optional are
        connected and configured.

        """
        allConfigured = self._initialized
        for slot in list(self.inputs.values()):
            allConfigured &= slot.ready() or slot._optional
        return allConfigured

    def _setDefaultInputValues(self):
        for i in list(self.inputs.values()):
            if i.upstream_slot is None and i._value is None and i._defaultValue is not None:
                i.setValue(i._defaultValue)

    def _disconnect(self):
        """Disconnect our slots from their upstream partners (not
        their downstream ones) and recursively do the same to all our
        child operators.

        """
        for s in list(self.inputs.values()) + list(self.outputs.values()):
            s.disconnect()

        # We must do this after the previous loop is complete.
        # Resizing multislots in depth-first order triggers the OperatorWrapper resize chain...
        for s in list(self.inputs.values()) + list(self.outputs.values()):
            if s.level > 1:
                s.resize(0)

        for child in list(self._children.keys()):
            child._disconnect()

    #  FIXME: Unused function?
    def disconnectFromDownStreamPartners(self):
        for slot in list(self.inputs.values()) + list(self.outputs.values()):
            downstream_slots = list(slot.downstream_slots)
            for p in downstream_slots:
                p.disconnect()

    def _initCleanup(self):
        self._cleaningUp = True
        for child in list(self._children.keys()):
            child._initCleanup()

    def cleanUp(self):
        if not self._cleaningUp:
            self._initCleanup()
        if self._parent is not None:
            del self._parent._children[self]

        # Disconnect ourselves and all children
        self._disconnect()

        for s in list(self.inputs.values()) + list(self.outputs.values()):
            # See note about the externally_managed flag in Operator.__init__
            downstream_slots = list(
                p for p in s.downstream_slots if p.operator is not None and not p.operator.externally_managed
            )
            if len(downstream_slots) > 0:
                msg = (
                    "Cannot clean up this operator ({}): Slot '{}'"
                    " is still providing data to downstream"
                    " operators!\n".format(self.name, s.name)
                )
                for i, p in enumerate(s.downstream_slots):
                    msg += "Downstream Partner {}: {}.{}".format(i, p.operator.name, p.name)
                raise RuntimeError(msg)

        self._parent = None

        # Work with a copy of the child list
        # (since it will be modified with each iteration)
        children = set(self._children.keys())
        for child in children:
            child.cleanUp()

    def _incrementOperatorExecutionCount(self):
        assert self._executionCount >= 0, "BUG: How did the execution count get negative?"
        # We can't execute while the operator is in the middle of
        # setupOutputs
        with self._condition:
            while self._settingUp:
                self._condition.wait()
            self._executionCount += 1

    def _decrementOperatorExecutionCount(self):
        assert self._executionCount > 0, "BUG: Can't decrement the execution count below zero!"
        with self._condition:
            self._executionCount -= 1
            self._condition.notifyAll()

    def propagateDirty(self, slot, subindex, roi):
        """This method is called when an output of another operator on
        which this operators depends, i.e. to which it is connected gets
        invalid. The region of interest of the inputslot which is now
        dirty is specified in the key property, the input slot which got
        dirty is specified in the inputSlot property.

        This method must calculate what output ports and which subregions
        of them are invalidated by this, and must call the .setDirty(key)
        of the corresponding outputslots.

        """
        raise NotImplementedError(".propagateDirty() of Operator {}" " is not implemented !".format(self.name))

    @staticmethod
    def forbidParallelExecute(func):
        """Use this decorator with functions that must not be run in
        parallel with the execute() function.

        - Your function won't start until no threads are in execute().

        - Calls to execute() will wait until your function is complete.

        This is better than using a simple lock in execute() because
        it allows execute() to be run in parallel with itself.

        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with self._condition:
                while self._executionCount > 0:
                    self._condition.wait()
                self._settingUp = True

                try:
                    return func(self, *args, **kwargs)
                finally:
                    self._settingUp = False
                    self._condition.notifyAll()

        wrapper.__wrapped__ = func  # Emulate python 3 behavior of @wraps
        return wrapper

    def _setupOutputs(self):
        # Don't setup this operator if there are currently
        # requests on it.
        if not self.configured():
            return

        with self._condition:
            while self._executionCount > 0:
                self._condition.wait()

            self._settingUp = True

            # Keep a copy of the old metadata for comparison.
            #  We only trigger downstream changes if something really changed.
            old_metadata = {s: s.meta.copy() for s in list(self.outputs.values())}

            # Call the subclass
            self.setupOutputs()
            self._setup_count += 1

            self._settingUp = False
            self._condition.notifyAll()

        try:
            # Determine new "ready" flags
            for k, oslot in list(self.outputs.items()):
                if oslot.upstream_slot is None:
                    # Special case, operators can flag an output as not actually being ready yet,
                    #  in which case we do NOT notify downstream connections.
                    if oslot.meta.NOTREADY:
                        oslot.disconnect()  # Forces unready state
                    else:
                        # All unconnected outputs are ready after
                        # setupOutputs
                        oslot._setReady()
                else:
                    assert oslot.meta.NOTREADY is None, (
                        "The special NOTREADY setting can only be used for output "
                        "slots that have no explicit upstream connection."
                    )

            # notify outputs of probably changed meta information
            for oslot in list(self.outputs.values()):
                if old_metadata[oslot] != oslot.meta and not (  # No need to call _changed() if nothing changed...
                    not old_metadata[oslot]._ready and oslot.meta._ready
                ):  # No need to call _changed() if it was already called in _setReady() above.
                    oslot._changed()
        except:
            # Something went wrong
            # Make the operator-supplied outputs unready again
            for k, oslot in list(self.outputs.items()):
                if oslot.upstream_slot is None:
                    oslot.disconnect()  # Forces unready state
            raise

    def handleInputBecameUnready(self, slot):
        # One of our input slots was disconnected.
        # If it was optional, we don't care.
        if slot._optional:
            return

        newly_unready_slots = []

        def set_output_unready(s):
            for ss in s._subSlots:
                set_output_unready(ss)
            if s.upstream_slot is None and s._value is None:
                was_ready = s.meta._ready
                s.meta._ready &= s.upstream_slot is not None
                if was_ready and not s.meta._ready:
                    newly_unready_slots.append(s)

        # All unconnected outputs are no longer ready
        for oslot in list(self.outputs.values()):
            set_output_unready(oslot)

        # If the ready status changed, signal it.
        for s in newly_unready_slots:
            s._sig_unready(s)
            s._changed()

    def setupOutputs(self):
        """This method is called when all input slots of an operator
        are successfully connected, a successful connection is also
        established if the input slot is not connected to another
        slot, but has a default value defined.

        In this method the operator developer should stup
        the .meta information of the outputslots.

        The default implementation emulates the old api behaviour.

        """
        for slot in list(self.outputs.values()):
            # This assert is here to force subclasses to override this method if the situation requires it.
            # If you have any output slots that aren't directly connected to an internal operator,
            #  you probably need to override this method.
            # If your subclass provides an implementation of this method, there
            #  is no need for it to call super().setupOutputs()
            assert slot.upstream_slot is not None, (
                "Output slot '{}' of operator '{}' has no upstream_slot, "
                "so you must override setupOutputs()".format(slot.name, self.name)
            )

    def call_execute(self, slot, subindex, roi, result, **kwargs):
        try:
            # We are executing the operator. Incremement the execution
            # count to protect against simultaneous setupOutputs()
            # calls.
            self._incrementOperatorExecutionCount()
            return self.execute(slot, subindex, roi, result, **kwargs)
        finally:
            self._decrementOperatorExecutionCount()

    def execute(self, slot, subindex, roi, result):
        """ This method of the operator is called when a connected
        operator or an outside user of the graph wants to retrieve the
        calculation results from the operator.

        The slot which is requested is specified in the slot arguemt,
        the region of interest is specified in the key property. The
        result area into which the calculation results MUST be written
        is specified in the result argument. "result" is an
        numpy.ndarray that has the same shape as the region of
        interest(key).

        The method must retrieve all required inputs that are
        neccessary to calculate the requested output area from its
        input slots, run the calculation and put the results into the
        provided result argument. """

        raise NotImplementedError("Operator {} does not implement" " execute()".format(self.name))

    def setInSlot(self, slot, subindex, roi, value):
        raise NotImplementedError(
            "Can't use __setitem__ with Operator {}" " because it doesn't implement" " setInSlot()".format(self.name)
        )

    @property
    def debug_text(self):
        # return self._debug_text
        return "setups: {}".format(self._setup_count)


#    @debug_text.setter
#    def debug_text(self, text):
#        self._debug_text = text


def format_operator_stack(tb):
    """
    Extract operator stacktrace from traceback
    """
    operator_stack = []
    for frame, lineno in walk_tb(tb):
        code = frame.f_code
        filename = code.co_filename
        locals_ = frame.f_locals
        mod_name = frame.f_globals["__name__"]

        maybe_op = locals_.get("self", None)
        if not isinstance(maybe_op, Operator):
            continue

        op_name = type(maybe_op).__qualname__
        qualname = f"{mod_name}.{op_name}.{code.co_name}"

        if op_name:
            operator_stack.append(FrameSummary(filename, lineno, qualname, lookup_line=False, locals=None))

    operator_stack.reverse()

    if operator_stack:
        return format_list(operator_stack)


_original_excepthook = sys.excepthook


def print_operator_stack(exc_type, exc, tb):
    """
    Enrich default exception output with operator stacktrace
    """
    _original_excepthook(exc_type, exc, tb)

    formatted = format_operator_stack(tb)
    if formatted:
        print("\n===Operator stack===\n", file=sys.stderr)
        print("".join(formatted), end="", file=sys.stderr)
        print("\n===Operator stack end===\n", file=sys.stderr)


def install_except_hook():
    sys.excepthook = print_operator_stack


def uninstall_except_hook():
    sys.excepthook = _original_excepthook


@contextmanager
def except_hook():
    install_except_hook()
    yield
    uninstall_except_hook()
