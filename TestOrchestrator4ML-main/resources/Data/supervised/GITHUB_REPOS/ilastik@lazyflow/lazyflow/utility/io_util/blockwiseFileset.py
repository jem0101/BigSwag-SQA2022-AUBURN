from future import standard_library

standard_library.install_aliases()
from builtins import zip
from builtins import map
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
import os
import copy
import shutil
import threading
import platform
import numpy
import h5py
import logging

logger = logging.getLogger(__name__)

import pickle as pickle

# The natural thing to do here is to use numpy.vectorize,
#  but that somehow interacts strangely with pickle.
# vectorized_pickle_dumps = numpy.vectorize( pickle.dumps, otypes=[str], 0 )
# vectorized_pickle_loads = numpy.vectorize( pickke.loads, otypes=[object] )


def vectorized_pickle_dumps(a):
    out = numpy.ndarray(shape=a.shape, dtype="O")
    for i, x in enumerate(a.flat):
        # Must use protocol 0 to avoid null bytes in the h5py dataset
        out.flat[i] = pickle.dumps(x, 0)
    return out


def vectorized_pickle_loads(a):
    out = numpy.ndarray(shape=a.shape, dtype=object)
    for i, s in enumerate(a.flat):
        out.flat[i] = pickle.loads(s)
    return out


from lazyflow.utility.jsonConfig import AutoEval, FormattedField, JsonConfigParser
from lazyflow.utility.log_exception import log_exception
from lazyflow.roi import getIntersection, roiToSlice
from lazyflow.utility import PathComponents, getPathVariants, FileLock
from lazyflow.roi import getIntersectingBlocks, getBlockBounds, TinyVector

try:
    import vigra

    _use_vigra = True
except:
    _use_vigra = False


class BlockwiseFilesetFactory(object):

    creationFns = set()

    @classmethod
    def register(cls, creationFn):
        BlockwiseFilesetFactory.creationFns.add(creationFn)

    @classmethod
    def create(cls, descriptionPath, mode):
        for fn in BlockwiseFilesetFactory.creationFns:
            bfs = fn(descriptionPath, mode)
            if bfs is not None:
                return bfs
        raise RuntimeError("Wasn't able to create a blockwise fileset from your file: {} ".format(descriptionPath))


class BlockwiseFileset(object):
    """
    This class handles writing and reading a 'blockwise file set'.
    A 'blockwise file set' is a directory with a particular structure, which contains the entire dataset broken up into blocks.
    Important parameters (e.g. shape, dtype, blockshape) are specified in a JSON file, which must match the schema given by :py:data:`BlockwiseFileset.DescriptionFields`.
    The parent directory of the description file is considered to be the top-most directory in the blockwise dataset hierarchy.

    - Simultaneous reads are threadsafe.
    - NOT threadsafe for reading and writing simultaneously (or writing and writing).
    - NOT threadsafe for closing.  Do not call close() while reading or writing.

    .. note:: See the unit tests in ``tests/testBlockwiseFileset.py`` for example usage.
    """

    #: These fields describe the schema of the description file.
    #: See the source code comments for a description of each field.
    DescriptionFields = {
        "_schema_name": "blockwise-fileset-description",
        "_schema_version": 1.1,
        "name": str,
        "format": str,
        "axes": str,
        "shape": AutoEval(numpy.array),  # This is the shape of the dataset on disk
        "dtype": AutoEval(),
        "drange": AutoEval(tuple),  # Optional. Data range, e.g. (0.0, 1.0)
        "chunks": AutoEval(numpy.array),  # Optional.  If null, no chunking. Only used when writing data.
        "compression": str,  # Optional.  Options include 'lzf' and 'gzip', among others.  Note: h5py automatically enables chunking on compressed datasets.
        "compression_opts": AutoEval(int),  # Optional. Hdf5-specific
        "block_shape": AutoEval(numpy.array),
        "view_origin": AutoEval(
            numpy.array
        ),  # Optional.  Defaults to zeros.  All requests will be translated before the data is accessed.
        # For example, if the offset is [100, 200, 300], then a request for roi([0,0,0],[2,2,2])
        #  will pull from the dataset on disk as though the request was ([100,200,300],[102,202,302]).
        # It is an error to specify an view_origin that is not a multiple of the block_shape.
        "view_shape": AutoEval(
            numpy.array
        ),  # Optional.  Defaults to (shape - view_origin) Limits the shape of the provided data.
        "block_file_name_format": FormattedField(
            requiredFields=["roiString"]
        ),  # For hdf5, include dataset name, e.g. myfile_block{roiString}.h5/volume/data
        "dataset_root_dir": str,  # Abs path or relative to the description file itself. Defaults to "." if left blank.
        "hash_id": str,  # Not user-defined (clients may use this)
        # Added in schema v1.1
        "sub_block_shape": AutoEval(numpy.array),  # Optional.  Must divide evenly into the block shape.
    }

    DescriptionSchema = JsonConfigParser(DescriptionFields)

    @classmethod
    def readDescription(cls, descriptionFilePath):
        """
        Parse the description file at the given path and return a
        :py:class:`jsonConfig.Namespace` object with the description parameters.
        The file will be parsed according to the schema given by :py:data:`BlockwiseFileset.DescriptionFields`.

        :param descriptionFilePath: The path to the description file to parse.
        """
        return BlockwiseFileset.DescriptionSchema.parseConfigFile(descriptionFilePath)

    @classmethod
    def writeDescription(cls, descriptionFilePath, descriptionFields):
        """
        Write a :py:class:`jsonConfig.Namespace` object to the given path.

        :param descriptionFilePath: The path to overwrite with the description fields.
        :param descriptionFields: The fields to write.
        """
        BlockwiseFileset.DescriptionSchema.writeConfigFile(descriptionFilePath, descriptionFields)

    class BlockNotReadyError(Exception):
        """
        This exception is raised if `readData()` is called for data that isn't available on disk.
        """

        def __init__(self, block_start):
            self.block_start = block_start

    @property
    def description(self):
        """
        The :py:class:`jsonConfig.Namespace` object that describes this dataset.
        """
        return self._description

    @classmethod
    def _createAndReturnBlockwiseFileset(self, descriptionFilePath, mode):
        try:
            bfs = BlockwiseFileset(descriptionFilePath, mode)
        except JsonConfigParser.SchemaError:
            bfs = None
        return bfs

    @classmethod
    def _prepare_system(cls):
        # None of this code is tested on Windows.
        # It might work, but you'll need to improve the unit tests to know for sure.
        assert (
            platform.system() != "Windows"
        ), "This code is all untested on Windows, and probably needs some modification before it will work."

        # If you get a "Too many open files" error, this soft limit may need to be increased.
        # The way to set this limit in bash is via "ulimit -n 4096"
        # Fortunately, Python lets us increase the limit via the resource module.
        import resource

        softlimit, hardlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        softlimit = max(4096, softlimit)
        resource.setrlimit(resource.RLIMIT_NOFILE, (softlimit, hardlimit))

    def __init__(self, descriptionFilePath, mode="r", preparsedDescription=None):
        """
        Constructor.  Uses `readDescription` interally.

        :param descriptionFilePath: The path to the .json file that describes the dataset.
        :param mode: Set to ``'r'`` if the fileset should be read-only.
        :param preparsedDescription: (Optional) Provide pre-parsed description fields, in which case the provided description file will not be parsed.
        """
        self._prepare_system()

        assert mode == "r" or mode == "a", "Valid modes are 'r' or 'a', not '{}'".format(mode)
        self.mode = mode

        assert (
            descriptionFilePath is not None
        ), "Must provide a path to the description file, even if you are providing pre-parsed fields. (Path is used to find block directory)."
        self._descriptionFilePath = descriptionFilePath

        if preparsedDescription is not None:
            self._description = preparsedDescription
        else:
            self._description = BlockwiseFileset.readDescription(descriptionFilePath)

        # Check for errors
        assert self._description.format == "hdf5", "Only hdf5 blockwise filesets are supported so far."
        if self._description.compression_opts is not None:
            assert (
                self._description.compression is not None
            ), "You specified compression_opts={} without specifying a compression type".format(
                self._description.compression
            )
        drange = self._description.drange
        if drange is not None:
            assert len(drange) == 2, "Invalid drange: {}".format(drange)
            assert drange[0] <= drange[1], "Invalid drange: {}".format(drange)

        sub_block_shape = self._description.sub_block_shape
        if sub_block_shape is not None:
            block_shape = self._description.block_shape
            block_shape_mods = numpy.mod(block_shape, sub_block_shape) != 0
            nonfull_block_shape_dims = block_shape != self._description.view_shape
            invalid_sub_block_dims = numpy.logical_and(nonfull_block_shape_dims, block_shape_mods)
            assert (invalid_sub_block_dims == False).all(), (
                "Each dimension of sub_block_shape must divide evenly into block_shape,"
                " unless the total dataset is only one block wide in that dimension."
            )

        # default view_origin
        if self._description.view_origin is None:
            self._description.view_origin = numpy.array((0,) * len(self._description.shape))
        assert (
            numpy.mod(self._description.view_origin, self._description.block_shape) == 0
        ).all(), "view_origin is not compatible with block_shape.  Must be a multiple!"

        # default view_shape
        if self._description.view_shape is None:
            self._description.view_shape = numpy.subtract(self._description.shape, self._description.view_origin)
        view_roi = (
            self._description.view_origin,
            numpy.add(self._description.view_origin, self._description.view_shape),
        )
        assert (
            numpy.subtract(self._description.shape, view_roi[1]) >= 0
        ).all(), "View ROI must not exceed on-disk shape: View roi: {}, on-disk shape: {}".format(
            view_roi, self._description.shape
        )

        if self._description.dataset_root_dir is None:
            # Default to same directory as the description file
            self._description.dataset_root_dir = "."

        self._lock = threading.Lock()
        self._openBlockFiles = {}
        self._fileLocks = {}
        self._closed = False

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """
        Close all open block files.
        """
        with self._lock:
            assert not self._closed
            paths = list(self._openBlockFiles.keys())
            for path in paths:
                blockFile = self._openBlockFiles[path]
                blockFile.close()
                if self.mode == "a":
                    fileLock = self._fileLocks[path]
                    fileLock.release()
            self._openBlockFiles = {}
            self._fileLocks = {}
            self._closed = True

    def reopen(self, mode):
        assert self._closed, "Can't reopen a fileset that isn't closed."
        self.mode = mode
        self._closed = False

    def readData(self, roi, out_array=None):
        """
        Read data from the fileset.

        :param roi: The region of interest to read from the dataset.  Must be a tuple of iterables: (start, stop).
        :param out_array: The location to store the read data.  Must be the correct size for the given roi.  If not provided, an array is created for you.
        :returns: The requested data.  If out_array was provided, returns out_array.
        """
        if out_array is None:
            out_array = numpy.ndarray(shape=numpy.subtract(roi[1], roi[0]), dtype=self._description.dtype)
        roi_shape = numpy.subtract(roi[1], roi[0])
        assert (roi_shape == out_array.shape).all(), "out_array must match roi shape"
        assert (roi_shape != 0).all(), "Requested roi {} has zero volume!".format(roi)
        self._transferData(roi, out_array, read=True)
        return out_array

    def writeData(self, roi, data):
        """
        Write data to the fileset.

        :param roi: The region of interest to write the data to.  Must be a tuple of iterables: (start, stop).
        :param data: The data to write.  Must be the correct size for the given roi.
        """
        assert self.mode != "r"
        assert (numpy.subtract(roi[1], roi[0]) != 0).all(), "Requested roi {} has zero volume!".format(roi)

        self._transferData(roi, data, read=False)

    def getDatasetDirectory(self, blockstart):
        """
        Return the directory that contains the block that starts at the given coordinates.
        """
        # Add the view origin to find the on-disk block coordinates
        blockstart = numpy.add(blockstart, self._description.view_origin)
        descriptionFileDir = os.path.split(self._descriptionFilePath)[0]
        absPath, _ = getPathVariants(self._description.dataset_root_dir, descriptionFileDir)
        blockFilePath = absPath

        for axis, start in zip(self._description.axes, blockstart):
            blockFilePath = os.path.join(blockFilePath, "{}_{:08d}".format(axis, start))
        return blockFilePath

    def _getBlockFileName(self, block_start):
        """
        Get the path to the block file that starts at the given coordinate.
        """
        # Translate to find disk block start
        block_start = numpy.add(self._description.view_origin, block_start)
        # Get true (disk) block bounds (i.e. use on-disk shape, not view_shape)
        entire_block_roi = getBlockBounds(self._description.shape, self._description.block_shape, block_start)
        roiString = "{}".format((list(entire_block_roi[0]), list(entire_block_roi[1])))
        datasetFilename = self._description.block_file_name_format.format(roiString=roiString)
        return datasetFilename

    def getDatasetPathComponents(self, block_start):
        """
        Return a PathComponents object for the block file that corresponds to the given block start coordinate.
        """
        datasetFilename = self._getBlockFileName(block_start)
        datasetDir = self.getDatasetDirectory(block_start)
        datasetPath = os.path.join(datasetDir, datasetFilename)

        return PathComponents(datasetPath)

    BLOCK_NOT_AVAILABLE = 0
    BLOCK_AVAILABLE = 1

    def getBlockStatus(self, blockstart):
        """
        Check a block's status.
        (Just because a block file exists doesn't mean that it has valid data.)
        Returns a status code of either ``BlockwiseFileset.BLOCK_AVAILABLE`` or ``BlockwiseFileset.BLOCK_NOT_AVAILABLE``.
        """
        blockDir = self.getDatasetDirectory(blockstart)
        statusFilePath = os.path.join(blockDir, "STATUS.txt")

        if not os.path.exists(statusFilePath):
            return BlockwiseFileset.BLOCK_NOT_AVAILABLE
        else:
            return BlockwiseFileset.BLOCK_AVAILABLE

    def isBlockLocked(self, blockstart):
        """
        Return True if the block is locked for writing.
        Note that both 'available' and 'not available' blocks might be locked.
        """
        datasetPathComponents = self.getDatasetPathComponents(blockstart)
        hdf5FilePath = datasetPathComponents.externalPath
        testLock = FileLock(hdf5FilePath)
        return not testLock.available()

    def setBlockStatus(self, blockstart, status):
        """
        Set a block status on disk.
        We use a simple convention: If the status file exists, the block is available.  Otherwise, it ain't.

        :param status: Must be either ``BlockwiseFileset.BLOCK_AVAILABLE`` or ``BlockwiseFileset.BLOCK_NOT_AVAILABLE``.
        """
        blockDir = self.getDatasetDirectory(blockstart)
        statusFilePath = os.path.join(blockDir, "STATUS.txt")

        if status == BlockwiseFileset.BLOCK_AVAILABLE:
            # touch the status file.
            open(statusFilePath, "w").close()
        elif os.path.exists(statusFilePath):
            # Remove the status file
            os.remove(statusFilePath)

    def setBlockStatusesForRoi(self, roi, status):
        block_starts = getIntersectingBlocks(self._description.block_shape, roi)
        for block_start in block_starts:
            self.setBlockStatus(block_start, status)

    def getEntireBlockRoi(self, block_start):
        """
        Return the roi for the entire block that starts at the given coordinate.
        """
        return getBlockBounds(self._description.view_shape, self._description.block_shape, block_start)

    def getAllBlockRois(self):
        """
        Return the list of rois for all VIEWED blocks in the dataset.
        """
        entire_dataset_roi = ([0] * len(self._description.view_shape), self._description.view_shape)
        block_starts = getIntersectingBlocks(self._description.block_shape, entire_dataset_roi)
        rois = []
        for block_start in block_starts:
            rois.append(self.getEntireBlockRoi(block_start))
        return rois

    def _transferData(self, roi, array_data, read):
        """
        Read or write data from/to the fileset.

        :param roi: The region of interest.
        :param array_data: If ``read`` is True, ``array_data`` is the destination array for the read data.  If ``read`` is False, array_data contains the data to write to disk.
        :param read: If True, read data from the fileset into ``array_data``.  Otherwise, write data from ``array_data`` into the fileset on disk.
        :type read: bool
        """
        entire_dataset_roi = ([0] * len(self._description.view_shape), self._description.view_shape)
        clipped_roi = getIntersection(roi, entire_dataset_roi)
        assert (
            numpy.array(clipped_roi) == numpy.array(roi)
        ).all(), "Roi {} does not fit within dataset bounds: {}".format(roi, self._description.view_shape)

        block_starts = getIntersectingBlocks(self._description.block_shape, roi)

        # TODO: Parallelize this loop?
        for block_start in block_starts:
            entire_block_roi = self.getEntireBlockRoi(block_start)  # Roi of this whole block within the whole dataset
            transfer_block_roi = getIntersection(
                entire_block_roi, roi
            )  # Roi of data needed from this block within the whole dataset
            block_relative_roi = (
                transfer_block_roi[0] - block_start,
                transfer_block_roi[1] - block_start,
            )  # Roi of needed data from this block, relative to the block itself
            array_data_roi = (
                transfer_block_roi[0] - roi[0],
                transfer_block_roi[1] - roi[0],
            )  # Roi of data needed from this block within array_data

            array_slicing = roiToSlice(*array_data_roi)
            self._transferBlockData(entire_block_roi, block_relative_roi, array_data, array_slicing, read)

    def _transferBlockData(self, entire_block_roi, block_relative_roi, array_data, array_slicing, read):
        """
        Read or write data to a single block in the fileset.

        :param entire_block_roi: The roi of the entire block, relative to the whole dataset.
        :param block_relative_roi: The roi of the data being read/written, relative to the block itself (not the whole dataset).
        :param array_data: Either the source or the destination of the data being transferred to/from the fileset on disk.
        :param read: If True, read data from the block into ``array_data``.  Otherwise, write data from ``array_data`` into the block on disk.
        :type read: bool
        """
        datasetPathComponents = self.getDatasetPathComponents(entire_block_roi[0])

        if self._description.format == "hdf5":
            self._transferBlockDataHdf5(
                entire_block_roi, block_relative_roi, array_data, array_slicing, read, datasetPathComponents
            )
        else:
            assert False, "Unknown format"

    def _transferBlockDataHdf5(
        self, entire_block_roi, block_relative_roi, array_data, array_slicing, read, datasetPathComponents
    ):
        """
        Transfer a block of data to/from an hdf5 dataset.
        See _transferBlockData() for details.

        We use separate parameters for array_data and array_slicing to allow users to pass an hdf5 dataset for array_data.
        """
        # For the hdf5 format, the full path format INCLUDES the dataset name, e.g. /path/to/myfile.h5/volume/data
        path_parts = datasetPathComponents
        datasetDir = path_parts.externalDirectory
        hdf5FilePath = path_parts.externalPath
        if len(path_parts.internalPath) == 0:
            raise RuntimeError(
                "Your hdf5 block filename format MUST specify an internal path, e.g. block{roiString}.h5/volume/blockdata"
            )

        block_start = entire_block_roi[0]
        if read:
            # Check for problems before reading.
            if self.getBlockStatus(block_start) is not BlockwiseFileset.BLOCK_AVAILABLE:
                raise BlockwiseFileset.BlockNotReadyError(block_start)

            hdf5File = self._getOpenHdf5Blockfile(hdf5FilePath)

            if (
                self._description.dtype != object
                and isinstance(array_data, numpy.ndarray)
                and array_data.flags.c_contiguous
            ):
                hdf5File[path_parts.internalPath].read_direct(
                    array_data, roiToSlice(*block_relative_roi), array_slicing
                )
            elif self._description.dtype == object:
                # We store arrays of dtype=object as arrays of pickle strings.
                array_pickled_data = hdf5File[path_parts.internalPath][roiToSlice(*block_relative_roi)]
                array_data[array_slicing] = vectorized_pickle_loads(array_pickled_data)
            else:
                array_data[array_slicing] = hdf5File[path_parts.internalPath][roiToSlice(*block_relative_roi)]

        else:
            # Create the directory
            if not os.path.exists(datasetDir):
                os.makedirs(datasetDir)
                # For debug purposes, output a copy of the settings
                #  that were active **when this block was created**
                descriptionFileName = os.path.split(self._descriptionFilePath)[1]
                debugDescriptionFileCopyPath = os.path.join(datasetDir, descriptionFileName)
                BlockwiseFileset.writeDescription(debugDescriptionFileCopyPath, self._description)

            # Clear the block status.
            # The CALLER is responsible for setting it again.
            self.setBlockStatus(block_start, BlockwiseFileset.BLOCK_NOT_AVAILABLE)

            # Write the block data file
            hdf5File = self._getOpenHdf5Blockfile(hdf5FilePath)
            if path_parts.internalPath not in hdf5File:
                self._createDatasetInFile(hdf5File, path_parts.internalPath, entire_block_roi)
            dataset = hdf5File[path_parts.internalPath]
            data = array_data[array_slicing]
            if data.dtype != object:
                dataset[roiToSlice(*block_relative_roi)] = data
            else:
                # hdf5 can't handle datasets with dtype=object,
                #  so we have to pickle each item first.
                pickled_data = vectorized_pickle_dumps(data)
                for index in numpy.ndindex(pickled_data.shape):
                    block_index = index + numpy.array(block_relative_roi[0])
                    dataset[tuple(block_index)] = list(pickled_data[index])

    def _createDatasetInFile(self, hdf5File, datasetName, roi):
        shape = tuple(roi[1] - roi[0])
        chunks = self._description.chunks
        if chunks is not None:
            # chunks must not be bigger than the data in any dim
            chunks = numpy.minimum(chunks, shape)
            chunks = tuple(chunks)
        compression = self._description.compression
        compression_opts = self._description.compression_opts

        dtype = self._description.dtype
        if dtype == object:
            dtype = h5py.special_dtype(vlen=numpy.uint8)
        dataset = hdf5File.create_dataset(
            datasetName,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Set data attributes
        if self._description.drange is not None:
            dataset.attrs["drange"] = self._description.drange
        if _use_vigra:
            dataset.attrs["axistags"] = vigra.defaultAxistags(str(self._description.axes)).toJSON()

    def _getOpenHdf5Blockfile(self, blockFilePath):
        """
        Return a handle to the open hdf5File at the given path.
        If we haven't opened the file yet, open it first.
        """
        # Try once without locking
        if blockFilePath in list(self._openBlockFiles.keys()):
            return self._openBlockFiles[blockFilePath]

        # Obtain the lock and try again
        with self._lock:
            if blockFilePath not in list(self._openBlockFiles.keys()):
                try:
                    writeLock = FileLock(blockFilePath, timeout=10)
                    if self.mode == "a":
                        acquired = writeLock.acquire(blocking=False)
                        assert acquired, "Couldn't obtain an exclusive lock for writing to file: {}".format(
                            blockFilePath
                        )
                        self._fileLocks[blockFilePath] = writeLock
                    elif self.mode == "r":
                        assert writeLock.available(), "Can't read from a file that is being written to elsewhere."
                    else:
                        assert False, "Unsupported mode"
                    self._openBlockFiles[blockFilePath] = h5py.File(blockFilePath, self.mode)
                except:
                    log_exception(logger, "Couldn't open {}".format(blockFilePath))
                    raise
            return self._openBlockFiles[blockFilePath]

    def getOpenHdf5FileForBlock(self, block_start):
        """
        Returns a handle to a file in this dataset.
        """
        block_start = tuple(block_start)
        path_components = self.getDatasetPathComponents(block_start)
        return self._getOpenHdf5Blockfile(path_components.externalPath)

    def purgeAllLocks(self):
        """
        Clears all .lock files from the local blockwise fileset.
        This may be necessary if previous processes crashed or were killed while some blocks were downloading.
        You must ensure that this is NOT called while more than one process (or thread) has access to the fileset.
        For example, in a master/worker situation, call this only from the master, before the workers have been started.
        """
        found_lock = False

        view_shape = self.description.view_shape
        view_roi = ([0] * len(view_shape), view_shape)
        block_starts = list(getIntersectingBlocks(self.description.block_shape, view_roi))
        for block_start in block_starts:
            blockFilePathComponents = self.getDatasetPathComponents(block_start)
            fileLock = FileLock(blockFilePathComponents.externalPath)
            found_lock |= fileLock.purge()
            if found_lock:
                logger.warning("Purged lock for block: {}".format(tuple(block_start)))

        return found_lock

    def exportRoiToHdf5(self, roi, exportDirectory, use_view_coordinates=True):
        """
        Export an arbitrary roi to a single hdf5 file.
        The file will be placed in the given exportDirectory,
        and will be named according to the exported roi.

        :param roi: The roi to export
        :param exportDirectory: The directory in which the result should be placed.
        :param use_view_coordinates: If True, assume the roi was given relative to the view start.
                                     Otherwise, assume it was given relative to the on-disk coordinates.
        """
        roi = list(map(TinyVector, roi))
        if not use_view_coordinates:
            abs_roi = roi
            assert (
                abs_roi[0] >= self.description.view_origin
            ), "Roi {} is out-of-bounds: must not span lower than the view origin: ".format(
                roi, self.description.origin
            )
            view_roi = roi - self.description.view_origin
        else:
            view_roi = roi
            abs_roi = view_roi + self.description.view_origin

        # Always name the file according to the absolute roi
        roiString = "{}".format((list(abs_roi[0]), list(abs_roi[1])))
        datasetPath = self._description.block_file_name_format.format(roiString=roiString)
        fullDatasetPath = os.path.join(exportDirectory, datasetPath)
        path_parts = PathComponents(fullDatasetPath)

        with h5py.File(path_parts.externalPath, "w") as f:
            self._createDatasetInFile(f, path_parts.internalPath, view_roi)
            dataset = f[path_parts.internalPath]
            self.readData(view_roi, dataset)

        return fullDatasetPath

    def exportSubset(self, roi, exportDirectory, use_view_coordinates=True):
        """
        Create a new blockwise fileset by copying a subset of this blockwise fileset.

        :param roi: The portion to export.  Must be along block boundaries, in ABSOLUTE coordinates.
        :param exportDirectory: The directory to copy the new blockwise fileset to.
        """
        # For now, this implementation assumes it can simply copy EVERYTHING in the block directories,
        #  including lock files.  Therefore, we require that the fileset be opened in read-only mode.
        # If that's a problem, change this function to ignore lock files when copying (or purge them afterwards).
        roi = list(map(TinyVector, roi))
        if not use_view_coordinates:
            abs_roi = roi
            assert (
                abs_roi[0] >= self.description.view_origin
            ), "Roi {} is out-of-bounds: must not span lower than the view origin: ".format(
                roi, self.description.origin
            )
        else:
            abs_roi = roi + self.description.view_origin

        assert self.mode == "r", "Can't export from a fileset that is open in read/write mode."

        block_shape = self._description.block_shape
        abs_shape = self._description.shape
        view_origin = self._description.view_origin

        assert (abs_roi[0] % block_shape == 0).all(), "exportSubset() requires roi to start on a block boundary"
        assert (
            (abs_roi[1] % block_shape == 0) | (abs_roi[1] == abs_shape)
        ).all(), "exported subset must end on block or dataset boundary."

        if not os.path.exists(exportDirectory):
            os.makedirs(exportDirectory)

        source_desc_path = self._descriptionFilePath
        source_desc_dir, source_desc_filename = os.path.split(source_desc_path)
        source_root_dir = self.description.dataset_root_dir

        # Copy/update description file
        dest_desc_path = os.path.join(exportDirectory, source_desc_filename)
        if os.path.exists(dest_desc_path):
            dest_description = BlockwiseFileset.readDescription(dest_desc_path)
        else:
            dest_description = copy.copy(self._description)
            dest_description.view_shape = abs_roi[1] - view_origin
            dest_description.hash_id = None

        BlockwiseFileset.writeDescription(dest_desc_path, dest_description)

        # Determine destination root block dir
        if os.path.isabs(source_root_dir):
            source_root_dir = os.path.normpath(source_root_dir)
            source_root_dir_name = os.path.split(source_root_dir)[1]
            dest_root_dir = os.path.join(exportDirectory, source_root_dir_name)
        else:
            dest_root_dir = os.path.join(exportDirectory, source_root_dir)

        source_root_dir, _ = getPathVariants(source_root_dir, source_desc_dir)

        view_roi = abs_roi - view_origin
        block_starts = getIntersectingBlocks(block_shape, view_roi)
        for block_start in block_starts:
            source_block_dir = self.getDatasetDirectory(block_start)
            rel_block_dir = os.path.relpath(source_block_dir, source_root_dir)
            dest_block_dir = os.path.join(dest_root_dir, rel_block_dir)

            if os.path.exists(dest_block_dir):
                logger.info("Skipping existing block directory: {}".format(dest_block_dir))
            elif not os.path.exists(source_block_dir):
                logger.info("Skipping missing block directory: {}".format(source_block_dir))
            else:
                # Copy the entire block directory
                assert dest_block_dir[-1] != "/"
                dest_block_dir_parent = os.path.split(dest_block_dir)[0]
                if not os.path.exists(dest_block_dir_parent):
                    os.makedirs(dest_block_dir_parent)
                shutil.copytree(source_block_dir, dest_block_dir)

        return dest_desc_path


BlockwiseFilesetFactory.register(BlockwiseFileset._createAndReturnBlockwiseFileset)
