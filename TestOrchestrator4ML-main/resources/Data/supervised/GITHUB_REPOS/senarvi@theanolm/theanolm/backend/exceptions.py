#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements custom exception classes.
"""

class InputError(Exception):
    """Exception raised for errors in the input.
    """
    pass

class NumberError(Exception):
    """Exception raised when one of the parameter gets NaN value.
    """
    pass

class IncompatibleStateError(Exception):
    """Exception raised when attempting to load a state that is incompatible
    with the neural network structure.
    """
    pass

class TheanoConfigurationError(Exception):
    """Exception raised when Theano has not been properly configured.
    """
    pass
