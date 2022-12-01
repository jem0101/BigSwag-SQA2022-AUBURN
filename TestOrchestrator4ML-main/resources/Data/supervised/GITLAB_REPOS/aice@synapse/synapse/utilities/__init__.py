"""
Copyright (C) 2019  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# region : Imports
import os
import abc
import importlib
# endregion : Imports


def check_initialize(func):
    def wrapper(cls, *args, **kwargs):
        assert cls._initialized, "Please initialize dataset : %s" % cls.name
        return func(cls, *args, **kwargs)
    return wrapper


def load_module(module_file_path):
    """dynamically load a module from the file path.

    Return: module object
    """

    if not os.path.exists(module_file_path):
        raise ValueError("Module loading error. {} is not exists.".format(module_file_path))

    head, _ = os.path.splitext(module_file_path)
    module_path = ".".join(head.split(os.sep))
    if os.path.isabs(module_file_path):
        spec = importlib.util.spec_from_file_location(module_path, module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return module


class BaseTrainer:
    """
    Base class for trainer object. It provides a templates to create custom trainer object.
    """
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def build_data_feeder(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_saver(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_summary(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_graph(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError
