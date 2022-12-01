"""
Methods used for parsing various configurations (dictionaries, json, yaml, etc.)

Attributes
----------
has_pyyaml : bool
    Whether the user has the PyYAML library installed (for parsing YAML files).
"""
# standard libraries
import logging
import collections
import os
import json
import copy
# third-party libraries
# check if pyyaml is installed
try:
    import yaml
    has_pyyaml = True
except ImportError as e:
    has_pyyaml = False

log = logging.getLogger(__name__)

def create_dictionary_like(input):
    """
    This takes in either an object or filename and parses it into a dictionary. Mostly useful for parsing JSON or YAML
    config files, and returning the dictionary representation.

    Parameters
    ----------
    input : collections.Mapping or str
        Dictionary-like object (implements collections.Mapping), JSON filename, or YAML filename.

    Returns
    -------
    collections.Mapping
        The parsed dictionary-like object, or None if it could not be parsed.

    .. note::

        YAML is parsed by the pyyaml library, which would be an optional dependency.
        Install with 'pip install pyyaml' if you want YAML-parsing capabilities.

    """
    if input is None:
        log.debug('Input to create_dictionary_like was None.')
        return None
    # check if it is a dictionary-like object (implements collections.Mapping)
    elif isinstance(input, collections.Mapping):
        return input
    # otherwise, check if it is a filename to a .json or .yaml
    elif os.path.isfile(input):
        _, extension = os.path.splitext(input)
        # if ends in .json
        if extension.lower() is '.json':
            with open(input, 'r') as json_data:
                return json.load(json_data)
        # if ends in .yaml
        elif (extension.lower() is '.yaml' or extension.lower() is '.yml') and has_pyyaml:
            with open(input, 'r') as yaml_data:
                return yaml.load(yaml_data)
        else:
            log.critical('Configuration file %s with extension %s not supported', str(input), extension)
            if not has_pyyaml:
                log.critical('Please install pyyaml with "pip install pyyaml" to parse yaml files.')
            return None
    # otherwise not recognized/supported:
    else:
        log.critical('Could not find config. Either was not collections.Mapping object or not found in filesystem.')
        return None

def combine_config_and_defaults(config=None, defaults=None):
    """
    This method takes two configuration dictionaries (or JSON/YAML filenames), and combines them.
    One will serve as the 'defaults' for the configuration, while the other will override any defaults when combined.

    Parameters
    ----------
    config : collections.Mapping or str
        Dictionary-like object or filepath to a JSON or YAML configuration file.
    defaults : collections.Mapping or str
        Dictionary-like object or filepath to a JSON or YAML configuration file.

    Returns
    -------
    collections.Mapping
        Dictionary-like object that combines the defaults and the config into one object, overriding any values
        found in defaults with values found in config.
    """
    # make sure the config is like a dictionary
    config_dict = copy.deepcopy(create_dictionary_like(config))
    # make sure the defaults is like a dictionary
    defaults_dict = copy.deepcopy(create_dictionary_like(defaults))
    # override any default values with the config (after making sure they parsed correctly)
    if config_dict is not None and defaults_dict is not None:
        defaults_dict.update(config_dict)
    # if there are no configuration options, give a warning because args will be None.
    elif config_dict is None and defaults_dict is None:
        log.warning("Both the config and defaults are None! Please supply at least one.")

    # set args to either the combined defaults and config, or just config if that is only one provided.
    if defaults_dict is not None:
        args = defaults_dict
    else:
        args = config_dict

    return args