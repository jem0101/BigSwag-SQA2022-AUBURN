from builtins import zip
from builtins import object
import sys

if sys.version_info.major >= 3:
    unicode = str

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
import json
import re
import collections
import numpy  # We import numpy here so that eval() understands names like "numpy.uint8"


class Namespace(object):
    """
    Provides the same functionality as:

    .. code-block:: python

        class Namespace(object):
            pass

    except that ``self.__dict__`` is replaced with an instance of collections.OrderedDict

    """

    def __init__(self):
        super(Namespace, self).__setattr__("_items", collections.OrderedDict())

    def __getattr__(self, key):
        items = super(Namespace, self).__getattribute__("_items")
        if key in items:
            return items[key]
        return super(Namespace, self).__getattribute__(key)

    def __setattr__(self, key, val):
        self._items[key] = val

    @property
    def __dict__(self):
        return self._items

    def __setstate__(self, state):
        """
        Implemented to support copy.copy()
        """
        super(Namespace, self).__setattr__("_items", collections.OrderedDict())
        self._items.update(state["_items"])

    def __eq__(self, other):
        """
        Compare two Namespace objects, with special treatment of numpy arrays to make sure they are compared correctly.
        """
        if not isinstance(other, Namespace):
            return False

        eq = True
        for (k1, v1), (k2, v2) in zip(list(self.__dict__.items()), list(other.__dict__.items())):
            eq &= k1 == k2
            if eq:
                b = v1 == v2
                if isinstance(b, numpy.ndarray):
                    eq &= b.all()
                else:
                    assert isinstance(b, bool)
                    eq &= b
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "jsonConfig.Namespace: " + str(list(self._items.items()))


class AutoEval(object):
    """
    Callable that serves as a pseudo-type.
    Converts a value to a specific type, unless the value is a string, in which case it is evaluated first.
    """

    def __init__(self, t=None):
        """
        If a type t is provided, the value from the config will be converted using t as the constructor.
        If t is not provided, the (possibly eval'd) value will be returned 'as-is' with no conversion.
        """
        self._t = t
        if t is None:
            # If no conversion type was provided, we'll assume that the result of eval() is good enough.
            self._t = lambda x: x

    def __call__(self, x):
        # Support these special type names without the need for a numpy prefix.
        from numpy import uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64

        if type(x) is self._t:
            return x
        if isinstance(x, (str, unicode)) and (not isinstance(self._t, type) or not issubclass(self._t, (str, unicode))):
            return self._t(eval(x))
        return self._t(x)


class FormattedField(object):
    """
    Callable that serves as a pseudo-type for config values that will be used by ilastik as format strings.
    Doesn't actually transform the given value, but does check it for the required format fields.
    """

    def __init__(self, requiredFields, optionalFields=[]):
        assert isinstance(requiredFields, list)
        assert isinstance(optionalFields, list)

        self._requiredFields = requiredFields
        self._optionalFields = optionalFields

    def __call__(self, x):
        """
        Convert x to str (no unicode), and check it for the required fields.
        """
        x = str(x)
        for f in self._requiredFields:
            fieldRegex = re.compile("{" + f + "(:[^}]*)?" + "}")
            if fieldRegex.search(x) is None:
                raise JsonConfigParser.ParsingError("Format string is missing required field: {{{f}}}".format(f=f))

        # TODO: Also validate that all format fields the user provided are known required/optional fields.
        return x


class RoiTuple(object):
    """
    Callale that serves as a pseudo-type.
    Converts a nested list to a roi tuple.
    """

    def __call__(self, x):
        if (
            not isinstance(x, (list, tuple))
            or len(x) != 2
            or not isinstance(x[0], (list, tuple))
            or not isinstance(x[1], (list, tuple))
            or len(x[0]) != len(x[1])
        ):
            raise JsonConfigParser.ParsingError("json value is not a valid roi: {}".format(x))

        # Are all values ints?
        for a in x[0] + x[1]:
            if not isinstance(a, int):
                raise JsonConfigParser.ParsingError("roi contains non-integers: {}".format(x))

        return (tuple(x[0]), tuple(x[1]))


class JsonConfigEncoder(json.JSONEncoder):
    """
    This special Json encoder standardizes the way that special types are written to JSON format.
    (e.g. numpy types, Namespace objects)
    """

    def default(self, o):
        import numpy

        if isinstance(o, numpy.integer):
            return int(o)
        if isinstance(o, numpy.floating):
            return float(o)
        if isinstance(o, numpy.ndarray):
            assert len(o.shape) == 1, "No support for encoding multi-dimensional arrays in json."
            return list(o)
        if isinstance(o, Namespace):
            return o.__dict__
        if isinstance(o, type):
            return o.__name__
        return super(JsonConfigEncoder, self).default(o)


class JsonConfigParser(object):
    """
    Parser for json config files that match a specific schema.
    Currently, only a very small set of json is supported.
    The schema fields must be a dictionary of name : type (or pseudo-type) pairs.

    A schema dict is also allowed as a pseudo-type value, which permits nested schemas.

    >>> # Specify schema as a dict
    >>> SchemaFields = {
    ...
    ...   "_schema_name" : "example-schema",
    ...   "_schema_version" : 1.0,
    ...
    ...   "shoe size" : int,
    ...   "color" : str
    ... }
    >>>
    >>> # Write a config file to disk for this example.
    >>> example_file_str = \\
    ... \"""
    ... {
    ...   "_schema_name" : "example-schema",
    ...   "_schema_version" : 1.0,
    ...
    ...   "shoe size" : 12,
    ...   "color" : "red",
    ...   "ignored_field" : "Fields that are unrecognized by the schema are ignored."
    ... }
    ... \"""
    >>> import os, tempfile
    >>> configFile = tempfile.NamedTemporaryFile(mode='w', delete=False)
    >>> _ = configFile.write(example_file_str)
    >>> configFile.close()
    >>>
    >>> # Create a parser that understands your schema
    >>> parser = JsonConfigParser( SchemaFields )
    >>>
    >>> # Parse the config file
    >>> parsedFields = parser.parseConfigFile(configFile.name)
    >>> os.remove(configFile.name)
    >>> print(parsedFields.color)
    red
    >>> # Whitespace in field names is replaced with underscores in the Namespace member.
    >>> print(parsedFields.shoe_size)
    12
    """

    class ParsingError(Exception):
        pass

    class SchemaError(ParsingError):
        pass

    def __init__(self, fields):
        self._fields = dict(fields)
        assert "_schema_name" in list(fields.keys()), "JsonConfig Schema must have a field called '_schema_name'"
        assert "_schema_version" in list(fields.keys()), "JsonConfig Schema must have a field called '_schema_version'"

        # Special case for the required schema fields
        self._requiredSchemaName = self._fields["_schema_name"]
        self._expectedSchemaVersion = self._fields["_schema_version"]

        self._fields["_schema_name"] = str
        self._fields["_schema_version"] = float

    def parseConfigFile(self, configFilePath):
        """
        Parse the JSON file at the given path into a :py:class:`Namespace` object that provides easy access to the config contents.
        Fields are converted from default JSON types into the types specified by the schema.
        """
        with open(configFilePath) as configFile:
            try:
                # Parse the json.
                # Use a special object_pairs_hook to preserve the user's field order and do some error checking, too.
                jsonDict = json.load(configFile, object_pairs_hook=self._createOrderedDictWithoutRepeats)
            except JsonConfigParser.ParsingError:
                raise
            except:
                import sys

                sys.stderr.write(
                    "File '{}' is not valid json.  See stdout for exception details.".format(configFilePath)
                )
                raise

            try:
                # Convert the dict we got into a namespace
                namespace = self._getNamespace(jsonDict)
            except JsonConfigParser.ParsingError as e:
                raise type(e)("Error parsing config file '{f}':\n{msg}".format(f=configFilePath, msg=e.args[0]))

        return namespace

    def writeConfigFile(self, configFilePath, configNamespace):
        """
        Simply write the given object to a json file as a dict,
        but check it for errors first by parsing each field with the schema.
        """
        # Check for errors by parsing the fields
        tmp = self._getNamespace(configNamespace.__dict__)

        with open(configFilePath, "w") as configFile:
            json.dump(configNamespace.__dict__, configFile, indent=4, cls=JsonConfigEncoder)

    def __call__(self, x):
        """
        This converts the given value (a dict) into a Namespace object.
        By implmenenting __call__ this way, we allow NESTED JsonConfigs.
        """
        try:
            namespace = self._getNamespace(x)
        except JsonConfigParser.ParsingError as e:
            raise type(e)("Couldn't parse sub-config:\n{msg}".format(msg=e.args[0]))
        return namespace

    def _getNamespace(self, jsonDict):
        """
        Convert the given dict into a Namespace object.
        Each value is transformed into the type given by the schema fields.
        """
        if isinstance(jsonDict, Namespace):
            jsonDict = jsonDict.__dict__
        if not isinstance(jsonDict, collections.OrderedDict):
            raise JsonConfigParser.ParsingError("Expected a collections.OrderedDict, got a {}".format(type(jsonDict)))
        configDict = collections.OrderedDict((str(k), v) for k, v in jsonDict.items())

        namespace = Namespace()
        # Keys that the user gave us are
        for key, value in configDict.items():
            if key in self._fields.keys():
                fieldType = self._fields[key]
                try:
                    finalValue = self._transformValue(fieldType, value)
                except JsonConfigParser.ParsingError as e:
                    raise type(e)("Error parsing config field '{f}':\n{msg}".format(f=key, msg=e.args[0]))
                else:
                    key = key.replace(" ", "_")
                    setattr(namespace, key, finalValue)

        # All other config fields are None by default
        for key in self._fields.keys():
            key = key.replace(" ", "_")
            if key not in namespace.__dict__.keys():
                setattr(namespace, key, None)

        # Check for schema errors
        if namespace._schema_name != self._requiredSchemaName:
            msg = "File schema '{}' does not match required schema '{}'".format(
                namespace._schema_name, self._requiredSchemaName
            )
            raise JsonConfigParser.SchemaError(msg)

        # Schema versions with the same integer (not fraction) are considered backwards compatible,
        # but not forwards-compatible.  For example:
        #     - expected 1.1, got 1.1 --> okay
        #     - expected 1.1, got 1.0 --> also okay
        #     - expected 1.1, got 1.2 --> error (can't understand versions from the future)
        #     - expected 1.1, got 0.9 --> error (integer changed, not backwards compatible)
        if namespace._schema_version > self._expectedSchemaVersion or int(namespace._schema_version) < int(
            self._expectedSchemaVersion
        ):
            msg = "File schema version '{}' is not compatible with expected schema version '{}'".format(
                namespace._schema_version, self._expectedSchemaVersion
            )
            raise JsonConfigParser.SchemaError(msg)

        return namespace

    def _transformValue(self, fieldType, val):
        """
        Convert val into the type given by fieldType.  Check for special cases first.
        """
        # config file is allowed to contain null values, in which case the value is set to None
        if val is None:
            return None

        # Check special error cases
        if fieldType is bool and not isinstance(val, bool):
            raise JsonConfigParser.ParsingError("Expected bool, got {}".format(type(val)))

        # Other special types will error check when they construct.
        return fieldType(val)

    def _createOrderedDictWithoutRepeats(self, pairList):
        """
        Used as the ``object_pairs_hook`` when parsing a json file.
        Creates an instance of collections.OrderedDict, but raises an exception
        if there are any repeated keys in the list of pairs.
        We only care about keys that are actually part of the schema.
        Note: There are some cases where this would do the wrong thing for NESTED schemas, but they are quite pathological.
        """
        ordered_dict = collections.OrderedDict()
        for k, v in pairList:
            if k in list(ordered_dict.keys()) and k in list(self._fields.keys()):
                raise JsonConfigParser.ParsingError("Invalid config: Duplicate entries for key: {}".format(k))
            # Insert the item
            ordered_dict[k] = v
        return ordered_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()
