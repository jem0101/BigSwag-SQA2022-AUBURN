"""
This module is for interacting with outputs for :class:`Monitor` objects (i.e. send the output to a file or database).
"""
# standard libraries
import logging
import os
# third party
from six import string_types  # for basestring compatability
# internal
from opendeep.utils.file_ops import mkdir_p

log = logging.getLogger(__name__)


class OutService(object):
    """
    Basic template for an OutService - needs a write() method to send the value to its proper destination.
    """
    def write(self, value, subset):
        """
        Given a value and the train/valid/test subset, send the value to the appropriate location.

        Parameters
        ----------
        value : object
            The value to write in the service.
        subset : str
            The subset that the value was created from (i.e. "train", "valid", or "test").

        Raises
        ------
        NotImplementedError
            If the method hasn't been implemented for the class yet.
        """
        log.exception("write() not implemented for %s!" % str(type(self)))
        raise NotImplementedError("write() not implemented for %s!" % str(type(self)))


class FileService(OutService):
    """
    Defines an OutService to write output to a given file.

    Attributes
    ----------
    train_filename : str
        Location for the file to write outputs from training.
    valid_filename : str
        Location for the file to write outputs from validation.
    test_filename : str
        Location for the file to write outputs from testing.
    """
    def __init__(self, filename):
        """
        Initialize a FileService and create train, valid, and test filenames from the given base filename.
        If a file exists with the same name, delete it.

        Parameters
        ----------
        filename : str
            Base filepath to use for the train, valid, and test files.
        """
        assert isinstance(filename, string_types), "input filename needs to be a string, found %s" % str(type(filename))
        self.value_separator = os.linesep
        filename = os.path.realpath(filename)
        basedir = os.path.dirname(filename)
        mkdir_p(basedir)
        # create the appropriate train, valid, test versions of the file
        name = os.path.basename(filename)
        name, ext = os.path.splitext(name)
        self.train_filename = os.path.join(basedir, name+'_train'+ext)
        self.valid_filename = os.path.join(basedir, name+'_valid'+ext)
        self.test_filename  = os.path.join(basedir, name+'_test'+ext)
        # delete the files if they already exist
        if os.path.exists(self.train_filename):
            os.remove(self.train_filename)
        if os.path.exists(self.value_separator):
            os.remove(self.valid_filename)
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def write(self, value, subset):
        """
        Given a value and the train/valid/test subset indicator, append the value to the appropriate file.

        Parameters
        ----------
        value : object
            The value to append to the file.
        subset : str
            The subset that the value was created from (i.e. "train", "valid", or "test").
        """
        val_to_write = str(value) + self.value_separator
        subset = subset.lower()
        if subset == "train":
            with open(self.train_filename, 'ab') as f:
                f.write(val_to_write)
        elif subset == "valid":
            with open(self.valid_filename, 'ab') as f:
                f.write(val_to_write)
        elif subset == "test":
            with open(self.test_filename, 'ab') as f:
                f.write(val_to_write)

class PrintService(object):
    """
    Defines an OutService to print the outputs to standard output.
    """
    def __init__(self, monitor_name):
        """
        Initialize a PrintService

        Parameters
        ----------
        monitor_name : str
            Name to prepend to standard output when printing the values (should be the same as the monitor that this
            service will output to avoid any confusion).
        """
        self.monitor_name = monitor_name

    def write(self, value, subset):
        """
        Given a value and the train/valid/test subset, print the value to standard output.

        Parameters
        ----------
        value : object
            The value to write in the service.
        subset : str
            The subset that the value was created from (i.e. "train", "valid", or "test").
        """
        print("%s_%s: %f" % (self.monitor_name, subset, value))
