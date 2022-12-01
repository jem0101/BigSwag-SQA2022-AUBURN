"""
This module contains utils that are general and can't be grouped logically into the other opendeep.utils modules.
"""
# standard libraries
import logging
import functools
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
# third party libraries
import numpy
import theano
import theano.tensor as T
from theano.compile.sharedvalue import SharedVariable
# internal imports
from opendeep.utils.constructors import as_floatX

log = logging.getLogger(__name__)


def make_time_units_string(time):
    """
    This takes a time (in seconds) and converts it to an easy-to-read format with the appropriate units.

    Parameters
    ----------
    time : int
        The time to make into a string (in seconds). Normally from computing differences in time.time().

    Returns
    -------
    str
        An easy-to-read string representation of the time i.e. "x hours", "x minutes", etc.
    """
    # Show the time with appropriate units.
    if time < 1:
        return trunc(time*1000)+" milliseconds"
    elif time < 60:
        return trunc(time)+" seconds"
    elif time < 3600:
        return trunc(time/60)+" minutes"
    else:
        return trunc(time/3600)+" hours"
    
def raise_to_list(input):
    """
    This will take an input and raise it to a List (if applicable). It will preserve None values as None.

    Parameters
    ----------
    input : object
        The object to raise to a list.

    Returns
    -------
    list
        The object as a list, or None.
    """
    if input is None:
        return None
    elif isinstance(input, list):
        return input
    else:
        return [input]
    
def stack_and_shared(input):
    """
    This will take a list of input variables, turn them into theano shared variables, and return them stacked
    in a single tensor.

    Parameters
    ----------
    input : list or object
        List of input variables to stack into a single shared tensor.

    Returns
    -------
    tensor
        Symbolic tensor of the input variables stacked, or None if input was None.
    """
    if input is None:
        return None
    elif isinstance(input, list):
        shared_ins = []
        for _in in input:
            try:
                shared_ins.append(theano.shared(_in))
            except TypeError as _:
                shared_ins.append(_in)
        return T.stack(shared_ins)
    else:
        try:
            _output = [theano.shared(input)]
        except TypeError as _:
            _output = [input]
        return T.stack(_output)
    
def concatenate_list(input, axis=0):
    """
    This takes a list of tensors and concatenates them along the axis specified (0 by default)

    Parameters
    ----------
    input : list
        List of tensors.
    axis : int, optional
        Axis to concatenate along.

    Returns
    -------
    tensor
        The concatenated tensor, or None if input was None.
    """
    if input is None:
        return None
    elif isinstance(input, list):
        return T.concatenate(input, axis=axis)
    else:
        return input
    
    
def closest_to_square_factors(n):
    """
    This function finds the integer factors that are closest to the square root of a number.
    (Useful for finding the closest width/height of an image you want to make square)

    Parameters
    ----------
    n : int
        The number to find its closest-to-square root factors.

    Returns
    -------
    tuple
        The tuple of (factor1, factor2) that are closest to the square root.
    """
    test = numpy.ceil(numpy.sqrt(float(n)))
    while not (n/test).is_integer():
        test -= 1
    if test < 1:
        test = 1
    return int(test), int(n/test)

def get_shared_values(variables, borrow=False):
    """
    This will return the values from a list of shared variables.

    Parameters
    ----------
    variables : list
        The list of shared variables to grab values from.
    borrow : bool
        The borrow argument for theano shared variable's `get_value()` method.

    Returns
    -------
    list
        The list of values held by the shared variables.

    Raises
    ------
    AttributeError
        If there wasn't a `get_value()` method for a variable in the input list.
    """
    try:
        values = [variable.get_value(borrow=borrow) for variable in variables]
    except AttributeError as e:
        log.exception("Cannot get values, there was an AttributeError %s",
                      str(e))
        raise
    return values

def set_shared_values(variables, values, borrow=False):
    """
    This sets the shared variables' values from a list of variables to the values specified in a list

    Parameters
    ----------
    variables : list
        The list of shared variables to set values.
    values : list
        The list of values to set the shared variables to.
    borrow : bool
        The borrow argument for theano shared variable's set_value() method.

    Raises
    ------
    ValueError
        If the list of variables and the list of values are different lengths.
    AttributeError
        If no `set_value()` function for a variable in the input list.
    """
    # use the safe_zip wrapper to ensure the variables and values lists are of the same length
    for variable, value in safe_zip(variables, values):
        # make sure the variable and value have the same shape
        assert value.shape == variable.get_value().shape, \
            "Shape mismatch! Value had shape %s, expected %s" % (str(value.shape), str(variable.get_value().shape))
        try:
            variable.set_value(value, borrow=borrow)
        except AttributeError as e:
            log.exception("Cannot set values, there was an AttributeError %s",
                          str(e))
            raise

def get_expression_inputs(expression):
    """
    This returns a list of all the distinct inputs to a theano computation expression

    Parameters
    ----------
    expression : theano expression
        The expression to find the inputs to the computation graph.

    Returns
    -------
    generator
        Yield the inputs found for the computation.
    """
    if hasattr(expression, 'owner'):
        if hasattr(expression.owner, 'inputs'):
            for input in expression.owner.inputs:
                yield get_expression_inputs(input)
        else:
            yield expression

def numpy_one_hot(vector, n_classes=None):
    """
    Takes a vector of integers and creates a matrix of one-hot encodings

    Parameters
    ----------
    vector : numpy.ndarray
        The integers to convert to one-hot encoding.
    n_classes : int
        The number of possible classes. if none, it will grab the maximum value from the vector.

    Returns
    -------
    numpy.ndarray
        A matrix of the one-hot encodings of the input vector.
    """
    vector = numpy.asarray(vector)
    assert isinstance(vector, numpy.ndarray), "Input vector couldn't be made into numpy array."
    # check if input is vector
    assert vector.ndim == 1, "Dimension mismatch for input vector, found %d dimensions!" % vector.ndim
    assert numpy.min(vector) > -1, "Found negative numbers in the vector, need all elements to be >= 0."
    # if no number classes specified, grab it from the vector
    if n_classes is None:
        max = numpy.max(vector)
        n_classes = max + 1
    # create matrix of zeros
    one_hot = numpy.zeros(shape=(len(vector), n_classes), dtype=theano.config.floatX)
    # fill in ones at the indices of the vector elements
    for i, element in enumerate(vector):
        one_hot[i, element] = 1
    return one_hot

def add_kwargs_to_dict(kwargs, dictionary):
    """
    Helper function to recursively add nested kwargs (flatten them) to a dictionary.

    Parameters
    ----------
    kwargs : dict
        The dictionary of keyword arguments, could have nested 'kwargs'.
    dictionary : dict
        The dictionary to flatten the keywords and value into.

    Returns
    -------
    dict
        The flattened dictionary of keyword arguments.
    """
    # Recursively add any kwargs into the given dictionary.
    for arg, val in kwargs.items():
        if arg not in dictionary and arg != 'kwargs':
            dictionary[arg] = val
        # flatten kwargs if it was passed as a variable
        elif arg == 'kwargs':
            inner_kwargs = kwargs['kwargs']
            dictionary = add_kwargs_to_dict(inner_kwargs, dictionary)
    return dictionary

def trunc(input, length=8):
    """
    Casts the input to a string and cuts it off after `length` characters.

    Parameters
    ----------
    input : object
        The input to truncate. Must be able to convert to String.
    length : int, optional
        The length of the resulting string (number of characters).

    Returns
    -------
    str
        The appropriately truncated string representation of `input`.
    """
    return str(input)[:length]

def binarize(input, cutoff=0.5):
    """
    Elementwise converts the input to 0 or 1.
    If element >= `cutoff` : 1; otherwise : 0.

    Parameters
    ----------
    input : tensor or array
        The number, vector, matrix, or tensor to binarize.
    cutoff : float
        The threshold value between [0, 1].

    Returns
    -------
    tensor or numpy array
        The input converted to 0 or 1 and cast to float.
    """
    return as_floatX(input >= cutoff)

def safe_zip(*args):
    """
    Like zip, but ensures arguments are of same length.

    Parameters
    ----------
    *args
        Argument list to `zip`

    Returns
    -------
    list
        The zipped list of inputs.

    Raises
    ------
    ValueError
        If the length of any argument is different than the length of args[0].
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument[0] has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)

def compose(*functions):
    """
    A functional helper for dealing with function compositions. It ignores any None functions, and if all are None,
    it returns None.

    Parameters
    ----------
    *functions
        function arguments to compose

    Returns
    -------
    function or None
        The composition f(g(...)) functions.
    """
    # help from here: https://mathieularose.com/function-composition-in-python/
    def f_of_g(f, g):
        if f is not None and g is not None:
            return lambda x: f(g(x))
        elif f is not None:
            return f
        elif g is not None:
            return g
        else:
            return lambda x: x

    if any(functions):
        return functools.reduce(f_of_g, functions, lambda x: x)
    else:
        return None

def min_normalized_izip(*iterables):
    """
    A function to make sure the length of all iterables is the same and normalize to the minimum if not.

    Parameters
    ----------
    *iterables
        A list of iterable objects (most typically going to be minibatches)
    """
    for elems in zip(*iterables):
        min_len = min([elem.shape[0] if hasattr(elem, 'shape') else len(raise_to_list(elem)) for elem in elems])
        yield [elem[:min_len] for elem in elems]

def base_variables(expression):
    """
    A helper to find the base SharedVariables in a given expression.

    Parameters
    ----------
    expression : theano expression
        The computation graph to find the base SharedVariables

    Returns
    -------
    set(SharedVariable)
        The set of unique shared variables
    """
    variables = set()
    if isinstance(expression, SharedVariable):
        variables.add(expression)
        return variables
    elif hasattr(expression, 'owner') and expression.owner is not None:
        for input in expression.owner.inputs:
            variables.update(base_variables(input))
    return variables
