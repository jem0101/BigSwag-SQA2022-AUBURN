"""Base classes for multiframe imaging data."""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future.utils import iteritems, itervalues
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
from past.builtins import basestring
import collections
import warnings
import itertools as it
import os
import csv
from os.path import dirname, join, abspath
import pickle as pickle
from distutils.version import StrictVersion

import numpy as np
try:
    import h5py
except ImportError:
    h5py_available = False
else:
    h5py_available = StrictVersion(h5py.__version__) >= StrictVersion('2.2.1')

import sima
import sima.misc
from sima.misc import mkdir_p, most_recent_key, estimate_array_transform, \
    estimate_coordinate_transform
from sima.extract import extract_rois, save_extracted_signals
from sima.ROI import ROIList
from sima.segment import ca1pc
from future import standard_library
standard_library.install_aliases()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sima.misc.tifffile import imsave


class ImagingDataset(object):

    """A multiple sequence imaging dataset.

    Imaging data sets can be iterated over to generate sequences, which
    can in turn be iterated over to generate imaging frames.

    Examples
    --------
    >>> import sima # doctest: +ELLIPSIS
    ...
    >>> from sima.misc import example_data
    >>> dataset = sima.ImagingDataset.load(example_data())

    Datasets can be iterated over as follows:

    >>> for sequence in dataset:
    ...     for frame in sequence:
    ...         for plane in frame:
    ...             for row in plane:
    ...                 for column in row:
    ...                     for channel in column:
    ...                         pass

    Datasets can also be indexed and sliced.

    >>> dataset[0].num_sequences
    1
    >>> dataset[:, 0].num_frames
    1
    >>> dataset[:, 0].frame_shape == dataset.frame_shape
    True

    The resulting sliced datasets are not saved by default.

    Parameters
    ----------
    sequences : list of sima.Sequence
        Imaging sequences that can each be iterated over to yield
        the imaging data from each acquisition time.
    savedir : str
        The directory used to store the dataset. If the directory
        name does not end with .sima, then this extension will
        be appended.
    channel_names : list of str, optional
        Names for the channels. Defaults to ['0', '1', '2', ...].

    Attributes
    ----------
    num_sequences : int
        The number of sequences in the ImagingDataset.
    frame_shape : tuple of int
        The shape of each frame, in order
        (num_planes, num_rows, num_columns, num_channels).
    num_frames : int
        The total number of image frames in the ImagingDataset.
    ROIs : dict of (str, ROIList)
        The sets of ROIs saved with this ImagingDataset.
    time_averages : list of ndarray
        The time-averaged intensity for each channel.
    time_std : ndarray
        The standard deviation of the intensity for each channel.
        Useful for knowing how dynamic each pixel is.
    time_kurtosis : ndarray
        The kurtosis of the intensity for each channel.
        Better than std deviation for knowing how dynamic each pixel is.

    """

    def __init__(self, sequences, savedir, channel_names=None,
                 read_only=False):

        self._read_only = read_only
        if sequences is None:
            # Special case used to load an existing ImagingDataset
            if not savedir:
                raise Exception('Cannot initialize dataset without sequences '
                                'or a directory.')

            with open(join(savedir, 'dataset.pkl'), 'rb') as f:
                data = pickle.load(f)
            if 'sequences' in data:
                # 1.0.0-dev sets stored sequences in dataset.pkl. Without this
                # check a later call to ImagingDataset.sequences will fail.
                # Remove in future version.
                raise ImportError('Old 1.0.0-dev dataset')
            self._channel_names = data.pop('channel_names', None)
            self._savedir = savedir
            try:
                self._num_frames = data.pop('num_frames')
            except KeyError:
                pass
            try:
                self._frame_shape = data.pop('frame_shape')
            except KeyError:
                pass
            try:
                self._num_sequences = data.pop('num_sequences')
            except KeyError:
                pass
        elif all(isinstance(s, sima.Sequence) for s in sequences):
            self.savedir = savedir
            self.sequences = sequences
            if channel_names is None:
                self.channel_names = [
                    str(x) for x in range(self.frame_shape[-1])]
            else:
                self.channel_names = channel_names
        else:
            raise TypeError('ImagingDataset objects must be initialized '
                            'with a list of sequences.')

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return ImagingDataset([self.sequences[indices]], None)
        indices = list(indices)
        seq_indices = indices.pop(0)
        if isinstance(seq_indices, int):
            seq_indices = slice(seq_indices, seq_indices + 1)
        sequences = [seq[tuple(indices)] for seq in self.sequences][
            seq_indices]
        return ImagingDataset(
            sequences, None, channel_names=self.channel_names)

    @property
    def sequences(self):
        if not hasattr(self, '_sequences'):
            def unpack(sequence):
                """Parse a saved Sequence dictionary."""
                return sequence.pop('__class__')._from_dict(
                    sequence, self.savedir)
            with open(join(self.savedir, 'sequences.pkl'), 'rb') as f:
                sequences = pickle.load(f)
            self._sequences = [unpack(seq) for seq in sequences]
            if not np.all([seq.shape[1:] == self._sequences[0].shape[1:]
                           for seq in self._sequences]):
                raise ValueError(
                    'All sequences must have images of the same size ' +
                    'and the same number of channels.')

        return self._sequences

    @sequences.setter
    def sequences(self, sequences):
        try:
            del self._frame_shape
        except AttributeError:
            pass
        try:
            del self._num_frames
        except AttributeError:
            pass
        try:
            del self._num_sequences
        except AttributeError:
            pass
        try:
            del self._time_averages
        except AttributeError:
            pass
        # TODO: Delete time_averages.pkl? Is that too aggressive?
        self._sequences = sequences

    @property
    def num_sequences(self):
        if not hasattr(self, '_num_sequences'):
            self._num_sequences = len(self.sequences)
        return self._num_sequences

    @property
    def frame_shape(self):
        if not hasattr(self, '_frame_shape'):
            self._frame_shape = self.sequences[0].shape[1:]
        return self._frame_shape

    @property
    def num_frames(self):
        if not hasattr(self, '_num_frames'):
            self._num_frames = sum(len(c) for c in self)
        return self._num_frames

    @property
    def channel_names(self):
        return self._channel_names

    @channel_names.setter
    def channel_names(self, names):
        self._channel_names = [str(n) for n in names]
        if self.savedir is not None and not self._read_only:
            self.save()

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        if savedir is None:
            self._savedir = None
        elif hasattr(self, '_savedir') and savedir == self.savedir:
            return
        else:
            if hasattr(self, '_savedir'):
                orig_dir = self.savedir
            else:
                orig_dir = False
            savedir = abspath(savedir)
            if not savedir.endswith('.sima'):
                savedir += '.sima'
            os.makedirs(savedir)
            self._savedir = savedir
            if orig_dir:
                from shutil import copy2
                for f in os.listdir(orig_dir):
                    if f.endswith('.pkl'):
                        try:
                            copy2(os.path.join(orig_dir, f), self.savedir)
                        except IOError:
                            pass
            if self._read_only:
                self._read_only = False

    @property
    def time_averages(self):
        if hasattr(self, '_time_averages'):
            return self._time_averages
        if self.savedir is not None:
            try:
                with open(join(self.savedir, 'time_averages.pkl'),
                          'rb') as f:
                    time_averages = pickle.load(f)
            except IOError:
                pass
            else:
                # Older versions of SIMA saved time_averages as a list
                # of arrays instead of a single 4D (zyxc) array.
                # Make sure this is a numpy array and if not just re-calculate
                # the time averages.
                if isinstance(time_averages, np.ndarray):
                    self._time_averages = time_averages
                    return self._time_averages

        sums = np.zeros(self.frame_shape)
        counts = np.zeros(self.frame_shape)
        for frame in it.chain.from_iterable(self):
            sums += np.nan_to_num(frame)
            counts[np.isfinite(frame)] += 1
        averages = old_div(sums, counts)
        if self.savedir is not None and not self._read_only:
            with open(join(self.savedir, 'time_averages.pkl'), 'wb') as f:
                pickle.dump(averages, f, pickle.HIGHEST_PROTOCOL)
        self._time_averages = averages
        return self._time_averages

    @property
    def time_std(self):
        if hasattr(self, '_time_std'):
            return self._time_std
        if self.savedir is not None:
            try:
                with open(join(self.savedir, 'time_std.pkl'),
                          'rb') as f:
                    time_std = pickle.load(f)
            except IOError:
                pass
            else:
                # Same protocol as the averages. Make sure the
                # std is a single 4D (zyxc) array and if not just
                # re-calculate the time std.
                if isinstance(time_std, np.ndarray):
                    self._time_std = time_std
                    return self._time_std

        sums = np.zeros(self.frame_shape)
        sums_squares = np.zeros(self.frame_shape)
        counts = np.zeros(self.frame_shape)
        for frame in it.chain.from_iterable(self):
            sums += np.nan_to_num(frame)
            sums_squares += np.square(np.nan_to_num(frame))
            counts[np.isfinite(frame)] += 1
        means = old_div(sums, counts)
        mean_of_squares = old_div(sums_squares, counts)
        std = np.sqrt(mean_of_squares - np.square(means))
        if self.savedir is not None and not self._read_only:
            with open(join(self.savedir, 'time_std.pkl'), 'wb') as f:
                pickle.dump(std, f, pickle.HIGHEST_PROTOCOL)
        self._time_std = std
        return self._time_std

    @property
    def time_kurtosis(self):
        if hasattr(self, '_time_kurtosis'):
            return self._time_kurtosis
        if self.savedir is not None:
            try:
                with open(join(self.savedir, 'time_kurtosis.pkl'),
                          'rb') as f:
                    time_kurtosis = pickle.load(f)
            except IOError:
                pass
            else:
                # Same protocol as the averages. Make sure the
                # kurtosis is a single 4D (zyxc) array and if not just
                # re-calculate the time kurtosis.
                if isinstance(time_kurtosis, np.ndarray):
                    self._time_kurtosis = time_kurtosis
                    return self._time_kurtosis

        sums = np.zeros(self.frame_shape)
        counts = np.zeros(self.frame_shape)
        for frame in it.chain.from_iterable(self):
            sums += np.nan_to_num(frame)
            counts[np.isfinite(frame)] += 1
        means = old_div(sums, counts)
        # Now calculate the kurtosis. Kurtosis as defined here is actually
        # not the standard Wiki definition. This is merely the 4th moment
        # of the mean subtracted trace for each pixel.
        sums_fourthpower = np.zeros(self.frame_shape)
        for frame in it.chain.from_iterable(self):
            sums_fourthpower += np.power(np.nan_to_num(frame - means), 4)
        kurtosis = old_div(sums_fourthpower, counts)

        if self.savedir is not None and not self._read_only:
            with open(join(self.savedir, 'time_kurtosis.pkl'), 'wb') as f:
                pickle.dump(kurtosis, f, pickle.HIGHEST_PROTOCOL)
        self._time_kurtosis = kurtosis
        return self._time_kurtosis

    @property
    def ROIs(self):
        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                return {label: ROIList(**v)
                        for label, v in pickle.load(f).items()}
        except (IOError, pickle.UnpicklingError):
            return {}

    @classmethod
    def load(cls, path):
        """Load a saved ImagingDataset object."""
        try:
            return cls(None, path)
        except ImportError as error:
            if not (error.args[0].endswith('iterables') or
                    error.args[0].endswith("iterables'")):
                raise error
            from sima.misc.convert import _load_version0
            # Load a read-only copy of the converted dataset
            ds = _load_version0(path)
            ds._read_only = True
            return ds

    def _todict(self):
        """Returns the dataset as a dictionary, useful for saving"""
        return {'savedir': abspath(self.savedir),
                'channel_names': self.channel_names,
                'num_frames': self.num_frames,
                'frame_shape': self.frame_shape,
                'num_sequences': self.num_sequences,
                '__version__': sima.__version__}

    def add_ROIs(self, ROIs, label=None):
        """Add a set of ROIs to the ImagingDataset.

        Parameters
        ----------
        ROIs : ROIList
            The regions of interest (ROIs) to be added.
        label : str, optional
            The label associated with the ROIList. Defaults to using
            the timestamp as a label.

        Examples
        --------
        Import an ROIList from a zip file containing ROIs created
        with NIH ImageJ.

        >>> from sima.ROI import ROIList
        >>> from sima.misc import example_imagej_rois,example_data
        >>> from sima.imaging import ImagingDataset
        >>> dataset = ImagingDataset.load(example_data())
        >>> rois = ROIList.load(example_imagej_rois(), fmt='ImageJ')
        >>> dataset.add_ROIs(rois, 'from_ImageJ')

        """

        if self.savedir is None:
            raise Exception('Cannot add ROIs unless savedir is set.')
        ROIs.save(join(self.savedir, 'rois.pkl'), label)

    def import_transformed_ROIs(
            self, source_dataset, method='affine', source_channel=0,
            target_channel=0, source_label=None, target_label=None,
            anchor_label=None, copy_properties=True,
            pre_processing_method=None, pre_processing_kwargs=None,
            **method_kwargs):
        """Calculate a transformation that maps the source ImagingDataset onto
        this ImagingDataset, transforms the source ROIs by this mapping,
        and then imports them into this ImagingDataset.

        Parameters
        ----------
        source_dataset : ImagingDataset
            The ImagingDataset object from which ROIs are to be imported.  This
            dataset must be roughly of the same field-of-view as self in order
            to calculate an affine transformation.

        method : string, optional
            Method to use for transform calculation.

        source_channel : string or int, optional
            The channel of the source image from which to calculate an affine
            transformation, either an integer index or a string in
            source_dataset.channel_names.

        target_channel : string or int, optional
            The channel of the target image from which to calculate an affine
            transformation, either an integer index or a string in
            self.channel_names.

        source_label : string, optional
            The label of the ROIList to transform

        target_label : string, optional
            The label to assign the transformed ROIList

        anchor_label : string, optional
            If None, use automatic dataset registration.
            Otherwise, the label of the ROIList that contains a single ROI
            with vertices defining anchor points common to both datasets.

        copy_properties : bool, optional
            Copy the label, id, tags, and im_shape properties from the source
            ROIs to the transformed ROIs

        pre_processing_method: string, optional
            pre-processing step applied before image registration

        pre_processing_kwargs: dictionary, optional
            arguments for pre-processing of image

        **method_kwargs
            Additional arguments can be passed in specific to the particular
            method. For example, 'order' for a polynomial transform estimation.

        """
        source_channel = source_dataset._resolve_channel(source_channel)
        target_channel = self._resolve_channel(target_channel)
        if pre_processing_kwargs is None:
            pre_processing_kwargs = {}

        if pre_processing_method == 'ca1pc':
            source = ca1pc._processed_image_ca1pc(
                source_dataset, channel_idx=source_channel,
                **pre_processing_kwargs)
            target = ca1pc._processed_image_ca1pc(
                self, channel_idx=target_channel,
                **pre_processing_kwargs)
        else:
            source = source_dataset.time_averages[..., source_channel]
            target = self.time_averages[..., target_channel]

        if anchor_label is None:
            try:
                transforms = [estimate_array_transform(s, t, method=method)
                              for s, t in zip(source, target)]
            except ValueError:
                print('Auto transform not implemented for this method')
                return
        else:
            # Assume one ROI per plane
            transforms = []
            for plane_idx in range(self.frame_shape[0]):
                trg_coords = None
                for roi in self.ROIs[anchor_label]:
                    if roi.coords[0][0, 2] == plane_idx:
                        # Coords is a closed polygon, so the last coord and the
                        # first coord are identical, remove one copy
                        trg_coords = roi.coords[0][:-1, :2]
                        break
                if trg_coords is None:
                    transforms.append(None)
                    continue

                src_coords = None
                for roi in source_dataset.ROIs[anchor_label]:
                    if roi.coords[0][0, 2] == plane_idx:
                        src_coords = roi.coords[0][:-1, :2]
                        break
                if src_coords is None:
                    transforms.append(None)
                    continue

                assert len(src_coords) == len(trg_coords)

                mean_dists = []
                for shift in range(len(src_coords)):
                    points1 = src_coords
                    points2 = np.roll(trg_coords, shift, axis=0)
                    mean_dists.append(
                        np.sum([np.sqrt(np.sum((p1 - p2) ** 2))
                                for p1, p2 in zip(points1, points2)]))
                trg_coords = np.roll(
                    trg_coords, np.argmin(mean_dists), axis=0)

                if method == 'piecewise-affine':

                    whole_frame_transform = estimate_coordinate_transform(
                        src_coords, trg_coords, 'affine')

                    src_additional_coords = [
                        [-50, -50],
                        [-50, source_dataset.frame_shape[1] + 50],
                        [source_dataset.frame_shape[2] + 50, -50],
                        [source_dataset.frame_shape[2] + 50,
                         source_dataset.frame_shape[1] + 50]]
                    trg_additional_coords = whole_frame_transform(
                        src_additional_coords)

                    src_coords = np.vstack((src_coords, src_additional_coords))
                    trg_coords = np.vstack((trg_coords, trg_additional_coords))

                transforms.append(estimate_coordinate_transform(
                    src_coords, trg_coords, method, **method_kwargs))

            transform_check = [t is None for t in transforms]
            assert not all(transform_check)

            if any(transform_check):
                warnings.warn("Z-plane missing transform. Copying from " +
                              "adjacent plane, accuracy not guaranteed")
                # If any planes were missing an anchor set, copy transforms
                # from adjacent planes
                for idx in range(len(transforms) - 1):
                    if transforms[idx + 1] is None:
                        transforms[idx + 1] = transforms[idx]
                for idx in reversed(range(len(transforms) - 1)):
                    if transforms[idx] is None:
                        transforms[idx] = transforms[idx + 1]

        src_rois = source_dataset.ROIs
        if source_label is None:
            source_label = most_recent_key(src_rois)
        src_rois = src_rois[source_label]

        transformed_ROIs = src_rois.transform(
            transforms, im_shape=self.frame_shape[:3],
            copy_properties=copy_properties)
        self.add_ROIs(transformed_ROIs, label=target_label)

    def delete_ROIs(self, label):
        """Delete an ROI set from the rois.pkl file

        Removes the file if no sets left.

        Parameters
        ----------
        label : string
            The label of the ROI Set to remove from the rois.pkl file

        """

        try:
            with open(join(self.savedir, 'rois.pkl'), 'rb') as f:
                rois = pickle.load(f)
        except IOError:
            return

        try:
            rois.pop(label)
        except KeyError:
            pass
        else:
            if len(rois):
                with open(join(self.savedir, 'rois.pkl'), 'wb') as f:
                    pickle.dump(rois, f, pickle.HIGHEST_PROTOCOL)
            else:
                os.remove(join(self.savedir, 'rois.pkl'))

    def export_averages(self, filenames, fmt='TIFF16', scale_values=True,
                        projection_type='average'):
        """Save TIFF files with the time average of each channel.

        For datasets with multiple frames, the resulting TIFF files
        have multiple pages.

        Parameters
        ----------
        filenames : str or list of str
            A single (.h5) output filename, or a list of (.tif) output
            filenames with one per channel.
        fmt : {'TIFF8', 'TIFF16', 'HDF5'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.
        projection_type : {'average', 'std', 'kurtosis'}, optional
            Whether to take average z projection, the std z projection, or
            the kurtosis projection. std and kurtosis are useful for
            finding the dynamic pixels

        """

        if fmt == 'HDF5':
            if not isinstance(filenames, basestring):
                raise ValueError(
                    'A single filename must be passed for HDF5 format.')
        elif not len(filenames) == self.frame_shape[-1]:
            raise ValueError(
                "The number of filenames must equal the number of channels.")
        if fmt == 'HDF5':
            if not h5py_available:
                raise ImportError('h5py >= 2.2.1 required')
            f = h5py.File(filenames, 'w')
            if projection_type == 'average':
                im = self.time_averages
            elif projection_type == 'std':
                im = self.time_std
            elif projection_type == 'kurtosis':
                im = self.time_kurtosis
            else:
                raise ValueError(
                    "projection_type must be 'average', 'std' or 'kurtosis'")
            if scale_values:
                im = sima.misc.to16bit(im)
            else:
                im = im.astype('uint16')
            f.create_dataset(name='time_' + projection_type, data=im)
            for idx, label in enumerate(['z', 'y', 'x', 'c']):
                f['time_' + projection_type].dims[idx].label = label
            if self.channel_names is not None:
                f['time_' + projection_type].attrs['channel_names'] = [
                    np.string_(s) for s in self.channel_names]
                # Note: https://github.com/h5py/h5py/issues/289
            f.close()
        else:
            for chan, filename in enumerate(filenames):
                if projection_type == 'average':
                    im = self.time_averages[:, :, :, chan]
                elif projection_type == 'std':
                    im = self.time_std[:, :, :, chan]
                elif projection_type == 'kurtosis':
                    im = self.time_kurtosis[:, :, :, chan]
                if dirname(filename):
                    mkdir_p(dirname(filename))
                if fmt == 'TIFF8':
                    if scale_values:
                        out = sima.misc.to8bit(im)
                    else:
                        out = im.astype('uint8')
                elif fmt == 'TIFF16':
                    if scale_values:
                        out = sima.misc.to16bit(im)
                    else:
                        out = im.astype('uint16')
                else:
                    raise ValueError('Unrecognized format.')
                imsave(filename, out)

    def export_frames(self, filenames, fmt='TIFF16', fill_gaps=True,
                      scale_values=False, compression=None, interlace=False):
        """Export imaging data from the dataset.

        Parameters
        ----------
        filenames : list of list of list of string or list of string
                    or list of list of strings
            Path to the locations where the output files will be saved.
            If fmt is TIFF, filenames[i][j][k] is the path to the file
            for sequence i, plane j, channel k.  If fmt is 'HDF5', filenames[i]
            is the path to the file for the ith sequence. If fmt is TIFF and
            interlace is True, filenames[i][k] is the path to the file for
            sequence i, channel k.
        fmt : {'TIFF8', 'TIFF16', 'HDF5'}, optional
            The format of the output files. Defaults to 16-bit TIFF.
        fill_gaps : bool, optional
            Whether to fill in unobserved rows with data from adjacent frames.
            Defaults to True.
        scale_values : bool, optional
            Whether to scale the values to use the full range of the
            output format. Defaults to False.  Channels are scaled separately.
        compression : {None, 'gzip', 'lzf', 'szip'}, optional
            If not None and 'fmt' is 'HDF5', compress the data with the
            specified lossless compression filter. See h5py docs for details on
            each compression filter.
        interlace : bool, optional
            Whether to save multiplane data as interlaced images in one
            multipage TIFF file per sequence/channel. Defaults to False.

        """

        try:
            depth = np.array(filenames).ndim
        except:  # noqa: E722
            raise TypeError('Improperly formatted filenames')
        if (fmt in ['TIFF16', 'TIFF8']) and not depth == 3 and not interlace:
            raise TypeError('Improperly formatted filenames')
        if interlace and not depth == 2:
            raise TypeError('Improperly formatted filenames')
        if fmt == 'HDF5' and not np.array(filenames).ndim == 1:
            raise TypeError('Improperly formatted filenames')
        for sequence, fns in zip(self, filenames):
            sequence.export(
                fns, fmt=fmt, fill_gaps=fill_gaps,
                channel_names=self.channel_names, compression=compression,
                scale_values=scale_values, interlace=interlace)

    def export_signals(self, path, fmt='csv', channel=0, signals_label=None):
        """Export extracted signals to a file.

        Parameters
        ----------
        path : str
            The name of the file that will store the exported data.
        fmt : {'csv'}, optional
            The export format. Currently, only 'csv' export is available.
        channel : string or int, optional
            The channel from which to export signals, either an integer
            index or a string in self.channel_names.
        signals_label : str, optional
            The label of the extracted signal set to use. By default,
            the most recently extracted signals are used.

        """

        try:
            csvfile = open(path, 'w', newline='')
        except TypeError:  # Python 2
            csvfile = open(path, 'wb')
        try:
            try:
                writer = csv.writer(csvfile, delimiter='\t')
            except TypeError:  # Python 2
                writer = csv.writer(csvfile, delimiter=b'\t')
            signals = self.signals(channel)
            if signals_label is None:
                signals_label = most_recent_key(signals)
            rois = signals[signals_label]['rois']

            writer.writerow(['sequence', 'frame'] + [r['id'] for r in rois])
            writer.writerow(['', 'label'] + [r['label'] for r in rois])
            writer.writerow(
                ['', 'tags'] +
                [''.join(t + ',' for t in sorted(r['tags']))[:-1]
                 for r in rois]
            )
            for sequence_idx, sequence in enumerate(
                    signals[signals_label]['raw']):
                for frame_idx, frame in enumerate(sequence.T):
                    writer.writerow([sequence_idx, frame_idx] + frame.tolist())
        finally:
            csvfile.close()

    def extract(self, rois=None, signal_channel=0, label=None,
                remove_overlap=True, n_processes=1, demix_channel=None,
                save_summary=True):
        """Extracts imaging data from the current dataset using the
        supplied ROIs file.

        Parameters
        ----------
        rois : sima.ROI.ROIList, optional
            ROIList of rois to extract
        signal_channel : string or int, optional
            Channel containing the signal to be extracted, either an integer
            index or a name in self.channel_names.
        label : string or None, optional
            Text label to describe this extraction, if None defaults to a
            timestamp.
        remove_overlap : bool, optional
            If True, remove any pixels that overlap between masks.
        n_processes : int, optional
            Number of processes to farm out the extraction across. Should be
            at least 1 and at most one less then the number of CPUs in the
            computer. Defaults to 1.
        demix_channel : string or int, optional
            Channel to demix from the signal channel, either an integer index
            or a name in self.channel_names If None, do not demix signals.
        save_summary : bool, optional
            If True, additionally save a summary of the extracted ROIs.

        Return
        ------
        signals: dict
            The extracted signals along with parameters and values calculated
            during extraction.
            Contains the following keys:
                raw, demixed_raw : list of arrays
                    The raw/demixed extracted signal. List of length
                    num_sequences, each element is an array of shape
                    (num_ROIs, num_frames).
                mean_frame : array
                    Time-averaged mean frame of entire dataset.
                overlap : tuple of arrays
                    Tuple of (rows, cols) such that zip(*overlap) returns
                    row, col pairs of pixel coordinates that are in more than
                    one mask. Note: coordinates are for the **flattened**
                    image, so 'rows' is always 0s.
                signal_channel : int
                    The index of the channel that was extracted.
                rois : list of dict
                    All the ROIs used for the extraction with the order matched
                    to the order of the rows in 'raw'.
                    See sima.ROI.todict for details of dictionary format.
                timestamp : string
                    Date and time of extraction in '%Y-%m-%d-%Hh%Mm%Ss' format

        See also
        --------
        sima.ROI.ROI
        sima.ROI.ROIList

        """

        signal_channel = self._resolve_channel(signal_channel)
        demix_channel = self._resolve_channel(demix_channel)

        if rois is None:
            rois = self.ROIs[most_recent_key(self.ROIs)]
        if rois is None or not len(rois):
            raise Exception('Cannot extract dataset with no ROIs.')
        if self.savedir:
            return save_extracted_signals(
                self, rois, self.savedir, label, signal_channel=signal_channel,
                remove_overlap=remove_overlap, n_processes=n_processes,
                demix_channel=demix_channel, save_summary=save_summary
            )
        else:
            return extract_rois(self, rois, signal_channel, remove_overlap,
                                n_processes, demix_channel)

    def save(self, savedir=None):
        """Save the ImagingDataset to a file."""

        if savedir is None:
            savedir = self.savedir
        self.savedir = savedir

        if self._read_only:
            raise Exception('Cannot save read-only dataset.  Change savedir ' +
                            'to a new directory')
        # Keep this out side the with statement
        # If sequences haven't been loaded yet, need to read sequences.pkl
        sequences = [seq._todict(savedir) for seq in self.sequences]
        with open(join(savedir, 'sequences.pkl'), 'wb') as f:
            pickle.dump(sequences, f, pickle.HIGHEST_PROTOCOL)

        with open(join(savedir, 'dataset.pkl'), 'wb') as f:
            pickle.dump(self._todict(), f, pickle.HIGHEST_PROTOCOL)

    def segment(self, strategy, label=None, planes=None):
        """Segment an ImagingDataset to generate ROIs.

        Parameters
        ----------
        strategy : sima.segment.SegmentationStrategy
            The strategy for segmentation.
        label : str, optional
            Label to be associated with the segmented set of ROIs.
        planes : list of int
            List of the planes that are to be segmented.

        Returns
        -------
        ROIs : sima.ROI.ROIList
            The segmented regions of interest.

        """

        rois = strategy.segment(self)
        if self.savedir is not None:
            rois.save(join(self.savedir, 'rois.pkl'), label)
        return rois

    def signals(self, channel=0):
        """Return a dictionary of extracted signals

        Parameters
        ----------
        channel : string or int
            The channel to load signals for, either an integer index or a
            string in self.channel_names

        """

        channel = self._resolve_channel(channel)
        try:
            with open(join(self.savedir, 'signals_{}.pkl'.format(channel)),
                      'rb') as f:
                return pickle.load(f)
        except (IOError, pickle.UnpicklingError):
            return {}

    def infer_spikes(self, channel=0, label=None, gamma=None,
                     share_gamma=True, mode='correct', verbose=False):
        """Infer the most likely discretized spike train underlying a
        fluorescence trace.

        Parameters
        ----------
        channel : string or int, optional
            The channel to be used for spike inference.
        label : string or None, optional
            Text string indicating the signals from which spikes should be
            inferred. Defaults to the most recently extracted signals.
        gamma : float, optional
            Gamma is 1 - timestep/tau, where tau is the time constant of the
            AR(1) process.  If no value is given, then gamma is estimated from
            the data.
        share_gamma : bool, optional
            Whether to apply the same gamma estimate to all ROIs. Defaults to
            True.
        mode : {'correct', 'robust', 'psd'}, optional
            The method for estimating sigma. The 'robust' method overestimates
            the noise by assuming that gamma = 1. The 'psd' method estimates
            sigma from the PSD of the fluorescence data. Default: 'correct'.
        verbose : bool, optional
            Whether to print status updates. Default: False.

        Returns
        -------
        spikes : ndarray of float
            The inferred normalized spike count at each time-bin.  Values are
            normalized to the maximum value over all time-bins.
            Shape: (num_rois, num_timebins).
        fits : ndarray of float
            The inferred denoised fluorescence signal at each time-bin.
            Shape: (num_rois, num_timebins).
        parameters : dict of (str, ndarray of float)
            Dictionary with values for 'sigma', 'gamma', and 'baseline'.

        Notes
        -----
        We strongly recommend installing MOSEK (www.mosek.com; free for
        academic use) which greatly speeds up the inference.

        References
        ----------
        * Pnevmatikakis et al. 2015. Submitted (arXiv:1409.2903).
        * Machado et al. 2015. Submitted.
        * Vogelstein et al. 2010. Journal of Neurophysiology. 104(6):
          3691-3704.

        """

        channel = self._resolve_channel(channel)

        import sima.spikes
        all_signals = self.signals(channel)
        if label is None:
            label = most_recent_key(all_signals)
        signals = all_signals[label]

        # estimate gamma for all cells
        if mode == "psd":
            if share_gamma:
                mega_trace = np.concatenate(
                    [sigs for sigs in signals['raw'][0]])
                sigma = sima.spikes.estimate_sigma(mega_trace)
                gamma = sima.spikes.estimate_gamma(mega_trace, sigma)
                sigma = [sigma for _ in signals['raw'][0]]
                gamma = [gamma for _ in signals['raw'][0]]
            else:
                sigma = [sima.spikes.estimate_sigma(sigs[0])
                         for sigs in signals['raw'][0]]
                gamma = [sima.spikes.estimate_gamma(sigs[0], sigm)
                         for sigm, sigs in zip(sigma, signals['raw'][0])]
        else:
            gamma = [sima.spikes.estimate_parameters(sigs, gamma, sigma=0)[0]
                     for sigs in zip(*signals['raw'])]
            if share_gamma:
                gamma = np.median(gamma)

            # ensure that gamma is a list, one value per ROI
            if isinstance(gamma, float):
                gamma = [gamma for _ in signals['raw'][0]]

            # estimate sigma values
            sigma = [sima.spikes.estimate_parameters(sigs, g)[1]
                     for g, sigs in zip(gamma, zip(*signals['raw']))]

        # perform spike inference
        spikes, fits, parameters = [], [], []
        for seq_idx, seq_signals in enumerate(signals['raw']):
            spikes.append(np.zeros_like(seq_signals))
            fits.append(np.zeros_like(seq_signals))
            parameters.append(collections.defaultdict(list))
            for i, trace in enumerate(seq_signals):
                spikes[-1][i], fits[-1][i], p = sima.spikes.spike_inference(
                    trace, sigma[i], gamma[i], mode, verbose)
                for k, v in iteritems(p):
                    parameters[-1][k].append(v)
            for v in itervalues(parameters[-1]):
                assert len(v) == len(spikes[-1])
            parameters[-1] = dict(parameters[-1])

        if self.savedir:
            signals['spikes'] = spikes
            signals['spikes_fits'] = fits
            signals['spikes_params'] = parameters
            all_signals[label] = signals

            signals_filename = os.path.join(
                self.savedir,
                'signals_{}.pkl'.format(signals['signal_channel']))

            pickle.dump(all_signals,
                        open(signals_filename, 'wb'), pickle.HIGHEST_PROTOCOL)

        return spikes, fits, parameters

    def __str__(self):
        return '<ImagingDataset>'

    def __repr__(self):
        return (
            '<ImagingDataset: ' + 'num_sequences={n_sequences}, ' +
            'frame_shape={fsize}, num_frames={frames}>'
        ).format(n_sequences=self.num_sequences,
                 fsize=self.frame_shape, frames=self.num_frames)

    def __iter__(self):
        return self.sequences.__iter__()

    def _resolve_channel(self, chan):
        """Return the index corresponding to the channel."""
        return sima.misc.resolve_channels(chan, self.channel_names)
