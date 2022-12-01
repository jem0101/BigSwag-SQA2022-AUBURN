#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:58:19 2018

@author: rensmasselink
"""
import logging
from multiprocessing import Manager
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from classifier.dataprep import check_training_data_nodata
from classifier.train import ALGORITHM_DICT
from classifier.utils.general import get_available_model_args, progress, \
    multiprocess_function
UNSUPERVISED_LOGGER = logging.getLogger(__name__)

def make_input_array(rasters, windows, config_dict):
    """ Create an input array for unsupervised classification

    Function for creating the input array for the training of the
    unsupervised models. A train_size relative to the entire dataset can be
    set to reduce the amount of memory needed

       Args:
           rasters (list)      input rasters
           windows (list)      A list of rasterio Windows to use
           config_dict (dict)   The parameters for making an input array

        Returns:
            A data array to use in model training"""
    # Create a 2d array which can be passed into the classifier
    bandcount = []
    for raster in rasters:
        with rasterio.open(raster, 'r') as src_r:
            bandcount.append(src_r.count)
    # Random index for choosing the windows to use
    subset_windows = np.random.choice(
        np.array(windows)[:, 1],
        size=int(len(windows) * config_dict['us_trainfraction']),
        replace=False
    )
    n_windows = len(subset_windows)
    assert n_windows > 0, (
        "Not enough subset data. Please increase -us_train_size parameter"
    )

    UNSUPERVISED_LOGGER.info(
        "\nUsing %i windows for training",
        n_windows
    )
    with Manager() as manager:
        mp_list = manager.list()
        arglist = [[mp_list,
                    rasters,
                    window,
                    i,
                    n_windows] for i, window in enumerate(subset_windows)]
        multiprocess_function(
            get_window_data_for_us_train,
            arglist,
            ncpus=config_dict['app_threads'])
        data_array = np.concatenate(mp_list)

    #Check for missing values and impute if necessary
    data_array = check_training_data_nodata(data_array, config_dict)

    return data_array[~np.isnan(data_array).any(axis=1)]

def get_window_data_for_us_train(arglist):
    """Gets the pixel values for window and adds it to an MP list

        Args:
            arglist (list): List of arguments containing:
                mp_list (mp.list): list to append the values to. List is changed
                                   inplace and is not returned
                rasters (list): List of rasters to get data from
                window (rasterio.Window): Window to process
                i (int): Number of the window
                n_windows (int): total number of windows

    """
    mp_list, rasters, window, i, n_windows = arglist
    data_array = []
    with rasterio.open(rasters[0], 'r') as template:
        # Create a vrt with the window and add the data to the array
        for raster in rasters:
            with rasterio.open(raster, 'r') as src:
                with WarpedVRT(
                    src,
                    resampling=Resampling.bilinear,
                    meta=template.meta,
                    width=template.meta['width'],
                    height=template.meta['height'],
                    crs=template.crs,
                    transform=template.transform
                ) as vrt:
                    data_array.append(vrt.read(window=window))
    old_shape = np.shape(data_array)
    mp_list.append(
        np.reshape(
            data_array,
            (old_shape[0]*old_shape[1], old_shape[2]*old_shape[3])
        ).transpose()
    )
    progress(i, n_windows)


def train_kmeans(train_array, config_dict, algorithm_args):
    """Train the kmeans model

        Args:
            train_array (array) A data array with columns (bands) and rows (
            pixels)
            config_dict (dict): Configuration Dictionary
            algorithm_args (dict): Dictionary to check args for in model classes

        Returns:
            A dictionary conaining the algorithm name, the trained model and an
            empty label key to match the dicts from the supervised
            classifications
    """
    algorithm = ALGORITHM_DICT[config_dict['app_algorithm']]
    n_classes = config_dict['us_nclasses']

    UNSUPERVISED_LOGGER.info("Now Training Model. This might take some time...")
    model_algorithms_args = get_available_model_args(algorithm_args, algorithm)
    kmeans = algorithm(n_clusters=n_classes, **model_algorithms_args)
    kmeans.fit(train_array)
    return {'app_algorithm': config_dict['app_algorithm'],
            'model': kmeans,
            'labels': None}
