"""Raster Utilities"""
import itertools
import logging
import os
import re
import subprocess
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT

from classifier.settings import TMP_DIR
from classifier.utils.general import progress, multiprocess_function, \
    write_tifs, impute
from classifier.utils.vector import get_subset_of_polygons

UTILS_RASTER_LOGGER = logging.getLogger(__name__)


def rasterize_polygons(polygons, clump_file, out_dir,
                       config_dict, col_name='max'):
    """Rasterize the classified Polygons.

    Args:
        polygons (GeoDataFrame): Classified polygons
        clump_file (str): Segmentation file
        out_dir (str): Output directory
        config_dict (dict): Configuration dictionary
        col_name (str): Column name to rasterize
    """
    windows, meta = get_meta([clump_file],
                             config_dict['app_window'])
    nr_windows = len(windows)

    polygons[col_name] = polygons[col_name].astype(int)
    iterable = [[x,
                 polygons,
                 meta,
                 nr_windows,
                 clump_file] for x in windows]

    multiprocess_function(
        rasterize_window,
        iterable,
        config_dict['app_threads']
    )
    stitch_(
        os.path.join(out_dir, 'classification_segments'),
        TMP_DIR,
        meta
    )


def stitch_(out_ds, temp_dir, meta, dtype='Int32'):
    """
    Stitch all tifs together to a single tif
    Args:
        out_ds: output file
        temp_dir: The temp directory containing the tifs
        meta: rasterio meta dictionary for output
        dtype: the raster's dtype

    """
    filelist = [os.path.join(temp_dir, x) for x in os.listdir(temp_dir) if
                x.endswith('tif')]
    input_list = temp_dir+'input_list.txt'
    with open(input_list, 'w') as tif_list:
        for files in filelist:
            tif_list.write('{}\n'.format(files))
    cmd_pred = ('gdalbuildvrt'
                ' -srcnodata {} {}.vrt -q -input_file_list {}').format(
                    meta['dst_meta']['nodata'],
                    out_ds, input_list)
    subprocess.call(cmd_pred, shell=True)
    cmd_pred = ('gdalwarp -ot {} -q -multi -wo "NUM_THREADS=ALL_CPUS" -co '
                '"TILED=YES"  -co "COMPRESS=DEFLATE" -overwrite -srcnodata  {'
                '} {}.vrt {}.tif').format(dtype,
                                          meta['dst_meta']['nodata'],
                                          out_ds,
                                          out_ds)
    subprocess.call(cmd_pred, shell=True)
    os.remove(out_ds+'.vrt')


def stitch(out_prefix, meta):
    """
     Stitch the temporary files together
    Args:
        out_prefix (str): The output directory
        meta (dict): The rasterio meta dicts from source, destination and
                     dest_proba

    """
    # Find all directories that have chunks.
    dirs = os.listdir(TMP_DIR)
    for directories in dirs:
        if directories.startswith('class'):
            out_file = os.path.join(out_prefix, 'classification')
            dtype = 'Int32'
        elif directories.startswith('probab'):
            out_file = os.path.join(out_prefix, 'probability')
            dtype = 'Float32'
        else:
            out_file = os.path.join(out_prefix, directories)
            dtype = 'Float32'
        UTILS_RASTER_LOGGER.info("Now stitching %s", out_file)
        stitch_(out_file,
                os.path.join(TMP_DIR, directories),
                meta,
                dtype)


def iterate_rasters_for_data(window, first, f_data, data_array, **kwargs):
    """
    Iterate over all rasters to add data into a single array
    Args:
        window: rasterio window
        first: first raster in the list
        f_data: the data from the first raster
        data_array: the array where all raster data is aggregated
        **kwargs:
        rasters: list of rasters


    Returns:
        The gathered data (array) and array where this data array is valid

    """
    bcounter = first.count
    data_array[:bcounter] = f_data
    # If there is more than 1 raster
    if len(kwargs['rasters']) > 1:

        # Create a vrt with the window and add the data to the array
        for raster in kwargs['rasters'][1:]:
            with rasterio.open(raster, 'r') as src:
                with WarpedVRT(src, resampling=Resampling.bilinear,
                               width=first.width,
                               height=first.height,
                               transform=first.transform,
                               crs=first.crs) as vrt:
                    raster_data = vrt.read(window=window)
                    # Fill the array with this data
                    data_array[
                        bcounter:bcounter + vrt.count,
                        :raster_data.shape[1],
                        :raster_data.shape[2]] = \
                        raster_data[:, :data_array.shape[1],
                                    :data_array.shape[2]]
                    bcounter += vrt.count
    data_array[data_array == first.nodatavals[0]] = np.nan
    if isinstance(data_array, list):
        data_array = np.concatenate(data_array, axis=0)
    if kwargs['config_dict']['app_imputation'] and data_array.size > 0:
        data_array = impute(data_array, **kwargs)
    valid = np.isfinite(data_array).all(axis=0)
    data = data_array[:, valid].T
    return data, valid


def get_meta(rasters, window_size=1024):
    """ Get the metadata from the input file and create the windows list to
    iterate through

    Args:
        raster  input raster
        window_size The size of the windows that will be processed
            """

    first_raster = rasters[0]
    # Read the metadata of the first and return metadata for dst and dst_proba
    with rasterio.open(first_raster, 'r') as first:
        f_meta = first.meta
        meta = f_meta.copy()
        meta.update(dtype=rasterio.dtypes.int32, nodata=-9999, count=1,
                    driver='GTiff')
        # Create the windows
        col_offs = np.arange(0, first.width, window_size)
        row_offs = np.arange(0, first.height, window_size)
        window_tuples = [x + (window_size, window_size) for x in list(
            itertools.product(col_offs, row_offs))]
        winlist = [rasterio.windows.Window(*x) for x in window_tuples]
        windows = list(zip(range(len(winlist)), winlist))
        meta_proba = meta.copy()
        meta_proba.update(dtype=rasterio.dtypes.float32, nodata=-9999)
    bandcount = []
    for raster in rasters:
        with rasterio.open(raster, 'r') as src_r:
            bandcount.append(src_r.count)
    return windows, {'f_meta': f_meta, 'dst_meta': meta,
                     'dst_proba_meta': meta_proba,
                     'bandcount': np.sum(bandcount)}


def open_single_raster(raster_path, window=None):
    """Opens single raster and returns numpy array.

    Array shape: nbands x nrows x ncols.

    Args:
        raster_path(str): location of the raster file
        window (rasterio.Window): subset window

    returns:
        ndarray:    Raster data
        meta (dict): dictionary with raster metadata
    """

    with rasterio.open(raster_path, 'r') as raster:
        raster_array = raster.read(window=window)
        meta = raster.meta
        meta.update(
            width=raster_array.shape[2],
            height=raster_array.shape[1],
            transform=raster.window_transform(window))
        return raster_array, meta


def rasterize_window(arglist):
    """Rasterize subset of a geodataframe.

    Arguments are supplied as a single list because the MP function does not
    allow that.

    Args:
        arglist (list) List with function arguments
            windows (list): number of window and Rasterio Window
            polygons: Polygons to rasterize
            meta_whole (dict): Meta of the whole (input) raster file
            total_windows (int): Total number of windows

    """
    # Get the arguments from the list
    i, window = arglist[0]
    polygons = arglist[1]
    meta_whole = arglist[2]
    total_windows = arglist[3]
    clump_file = arglist[4]

    # Progress Bar
    progress(i, total_windows)

    _, meta = open_single_raster(clump_file, window=window)

    # Get subset of polygons for this window
    to_rasterize = get_subset_of_polygons(window,
                                          meta_whole['dst_meta']['transform'],
                                          polygons)
    # Get shape of output raster and rasterize
    out_shape = [meta['height'], meta['width']]
    rasterized = rasterize(to_rasterize,
                           fill=-9999,
                           transform=meta['transform'],
                           out_shape=out_shape).astype(np.int32)
    # Write
    meta['compress'] = 'deflate'
    meta['dtype'] = rasterio.int32
    write_tifs(
        TMP_DIR,
        window,
        meta,
        rasterized)


def get_raster_date(raster):
    """Gets the date of a tif file from the name. The pattern that is expected
    is YYYY-MM-DD.

    If there are multiple dates with that format in the filename, only the
    first one will be used

        Args:
            raster (str) : Path to raster file
        returns:
            file_date (str): date of the raster in string format
    """
    file_date = re.findall(r'\d\d\d\d-\d\d-\d\d', raster)[0]
    return file_date

def raster_warp_dst(raster):
    """Get the warp paramters of a raster:

        Args:
            raster(str): path to a raster file

        Returns:
            warp_dst (dict): warp parameter dict
    """
    with rasterio.open(raster) as first:
        warp_dst = {
            "width": first.width,
            "height": first.height,
            "transform": first.transform,
            "crs": first.crs
        }
    return warp_dst

def count_bands(raster):
    """Returns number of bands of a raster

        Arg:
            raster (str): path to raster file

    """
    with rasterio.open(raster, 'r') as src:
        return src.count

def verify_and_count_bands(rasters):
    """Counts the number of bands for all the rasters and makes sure they all
    have the same number bands. Then it returns the number of bands that the
    rasters have

        Args:
            rasters(list): list of raster files

        Returns:
            b_count (int): number of bands in each raster

    """
    n_bands_list = [count_bands(raster) for raster in rasters]
    count = set(n_bands_list)
    if len(count) > 1:
        UTILS_RASTER_LOGGER.error("Number of bands per raster is not equal. "
                                  "Quitting...")
        sys.exit()
    return count.pop()
