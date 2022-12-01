"""All timeseries functionality can be found here"""
import datetime
import logging
from multiprocessing import Manager

import fiona
import numpy as np
import pandas as pd

from classifier.dataprep import gather_samples_for_roi
from classifier.utils.general import multiprocess_function, impute_values, \
    progress
from classifier.utils.raster import get_raster_date, raster_warp_dst, \
    verify_and_count_bands

TIMESERIES_LOGGER = logging.getLogger(__name__)



def gather_samples_ts_per_roi(arg_list):
    """Gather samples for a timeseries for a roi, for usage in MP function

        Args:
            arg_list(list): list of arguments for mp function containing:
                mp_list (mp.List): list for use in multiprocessing
                roi (shapely.geometry): region of interest
                ts_dict(dict): dictionary of dates and rasters
                config_dict(dict): Configuration Dictionary
                warp_dst (dict): warp parameters for raster
                df_index (list): index to use for creation of samples df
                bands (list): bands to use from the tifs
                i (int): number of the job to use in progress bar
                total(int): total number of jobs for progress bar
    """
    mp_list, roi, ts_dict, config_dict, warp_dst, \
        df_index, bands, i, total = arg_list

    if total > 1:
        progress(i, total)
    roi_samples = gather_samples_for_roi(
        roi,
        ts_dict.values(),
        warp_dst
    )

    index = pd.MultiIndex.from_product(
        [df_index, bands],
        names=['Date', 'Band']
    )
    roi_id = int(roi['properties']['id'])
    roi_classes = roi_samples.shape[0] * [roi_id]
    last_df = pd.DataFrame(
        roi_samples.transpose(),
        index=index,
        columns=roi_classes)

    imputed_df = impute_timeseries(last_df, config_dict)
    mp_list.append(imputed_df)


def gather_samples_ts(ts_dict, config_dict, rois_file=None, roi=None):
    """Gathers the pixel values for all the timeseries rasters and
    combines them in a df

    The DataFrame is a multiColumn dataframe where column level 0 is the class

        Args:
            ts_dict (dict): timeseries raster dictionary
            config_dict(dict): Configuration dictionary
            rois_file (str): Path to the ogr rois file when doing sample
                             gathering for training a model
            roi (shapely.geometry: geometry when gathering pixel values when
                                    predicting a single chunk

        returns:
            samples_df (DataFrame): DataFrame with timeseries of samples
    """

    # Get pandas index
    df_index = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in ts_dict]
    # Iterate through the polygons and return
    rasters = list(ts_dict.values())
    warp_dst = raster_warp_dst(rasters[0])
    b_count = verify_and_count_bands(rasters)
    bands = ['B{:02d}'.format(x) for x in np.arange(1, b_count + 1)]
    with Manager() as manager:
        mp_list = manager.list()
        if rois_file is not None:
            with fiona.open(rois_file, "r") as shapefile:
                total_rois = len(shapefile)
                arg_lists = [
                    [mp_list,
                     roi,
                     ts_dict,
                     config_dict,
                     warp_dst,
                     df_index,
                     bands,
                     i,
                     total_rois] for
                    i, roi in enumerate(shapefile)]
        else:
            arg_lists = [[
                mp_list,
                roi,
                ts_dict,
                config_dict,
                warp_dst,
                df_index,
                bands,
                1,
                1
            ]]
        multiprocess_function(
            gather_samples_ts_per_roi,
            arg_lists,
            ncpus=config_dict['app_threads']
        )
        samples_df = pd.concat(mp_list, axis=1)
        samples_df.columns = samples_df.columns.astype(str)
        return samples_df


def impute_timeseries(samples_df, config_dict):
    """Imputation of timeseries

        Args:
            samples_df(pd.DataFrame): Dataframe with timeseries samples
            config_dict(dict): Configuration dicitionary

        Returns:
            data_filled_df(pd.DataFrame): Imputed DataFrame
    """
    if config_dict['app_imputation_strategy'] == 'interpolate':
        initial_columns = samples_df.columns
        samples_df.columns = [str(x) for x in range(len(samples_df.columns))]

        data_filled_df = samples_df.unstack(level=-1).interpolate(
            method='time',
            axis=0
            ).stack(dropna=False)
        data_filled_df.columns = initial_columns

        if data_filled_df.isnull().values.any():
            # If there are still any nodata pixels left...
            TIMESERIES_LOGGER.warning("Found pixels without any data. "
                                      "Setting mean chunk/roi values "
                                      "to the pixels")

            mean = data_filled_df.stack().mean()
            data_filled_df.fillna(mean, inplace=True)

    else:
        data_array = samples_df.values
        data_filled = [impute_values(data_array, config_dict)]
        data_filled_df = pd.DataFrame(
            np.concatenate(data_filled),
            index=samples_df.index,
            columns=samples_df.columns)
    return data_filled_df


def get_timeseries_samples(rasters, rois, out_dir, config_dict):
    """Samples for timeeseries from a list of rasters and a polygon file

        Args:
            rasters(list): list of raster files
            rois(str): Path to OGR polygon file
            out_dir(str): Path to save the samples to
            config_dict(dict): Configuration dictionary

        Returns:
            samples_df (pd.DataFrame): samples and classes
    """
    raster_dates = [get_raster_date(raster) for raster in rasters]
    raster_dict = dict(zip(raster_dates, rasters))
    samples_df = gather_samples_ts(raster_dict, rois, config_dict)
    if out_dir is not None:
        samples_df.to_csv('{}/samples_ts.csv'.format(out_dir))
    return samples_df
