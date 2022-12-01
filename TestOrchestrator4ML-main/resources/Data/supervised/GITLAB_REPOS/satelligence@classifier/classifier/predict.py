"Prediction Module"
import logging
from multiprocessing import process
import os

import numpy as np
import rasterio

from classifier.utils.raster import iterate_rasters_for_data
from classifier.utils.raster import write_tifs
from classifier.settings import TMP_DIR, US_ALGORITHMS
from classifier.timeseries import get_timeseries_samples

# need to set the temp dir for multiprocessing in a container
process.current_process()._config['tempdir'] = '/tmp/' #pylint ig

# Load all rasters in a windowed manner (in a vrt) and do the prediction
PREDICT_LOGGER = logging.getLogger(__name__)


def prediction_class_probabilities(predicted, valid, meta, window):
    """Gets probability for the classified class and writes the tif for a
    windows

    Args:
        predicted (np.ndarray): The probabilities for all classes
        valid (np.ndarray): Array with valid values for prediction
        meta (dict): rasterio meta for writing tif
        window: rasterio.Window to predict and write

    """
    labels_internal_proba = (np.nanmax(predicted, axis=1)
                             .astype(np.float32))
    result_proba = np.full(valid.shape, -9999, np.float32)
    result_proba[valid] = labels_internal_proba
    write_tifs(os.path.join(TMP_DIR, 'probability'),
               window,
               meta,
               result_proba)

def prediction_all_probabilities(predicted, valid, meta, window):
    """Gets probability for the classified class and writes the tif for a
    windows

    Args:
        predicted (np.ndarray): The probabilities for all classes
        valid (np.ndarray): Array with valid values for prediction
        meta (dict): rasterio meta for writing tif
        window: rasterio.Window to predict and write

    """
    # loop over the classes and save them all separately
    for internal_label in range(predicted.shape[1]):
        probabilities = predicted[:, internal_label]
        result_proba = np.full(valid.shape, -9999, np.float32)
        result_proba[valid] = probabilities
        write_tifs(os.path.join(TMP_DIR, str(internal_label)),
                   window,
                   meta,
                   result_proba)

def do_prediction_proba_su(data, valid, model_dict, config_dict, meta, window):
    """Does the actual prediction and probability description for supervised
    methods

    Args:
        data: Numpy array with features
        valid: Which parts of the numpy array have valid data
            model_dict:  The model dictionary containting algorithm info
            config_dict: The cli configuration
            meta: The rasterio meta dictionary for the outputs
            window: The rasterio Window for which to do the prediction

    Returns:
        The predictions array

    """
    predicted = model_dict['model'].predict_proba(data)
    # Get the final classification of the highest probabilities
    labels_external = np.nanargmax(predicted, axis=1)
    result = np.full(valid.shape, -9999, np.int32)
    result[valid] = labels_external

    # See if we need to output some of the modelled probabilities
    if config_dict['su_probability']:
        prediction_class_probabilities(predicted, valid, meta, window)
    if config_dict['su_all_probabilities']:
        prediction_all_probabilities(predicted, valid, meta, window)

    return result

def do_prediction_proba(data, valid, prediction_dict, window):
    """Calls the actual probability functions for (un)supervised
    methods

    Args:
        data: Numpy array with features
        valid: Which parts of the numpy array have valid data
            model_dict:  The model dictionary containting algorithm info
            config_dict: The cli configuration
            meta: The rasterio meta dictionary for the outputs
            window: The rasterio Window for which to do the prediction

    Returns:
        The probability array

    """
    model_dict = prediction_dict['model_dict']
    config_dict = prediction_dict['config_dict']
    meta = prediction_dict['meta']['dst_proba_meta']
    if config_dict['app_algorithm'] == 'singleclass':
        result = predict_single_class(
            data,
            valid,
            model_dict,
            config_dict,
            meta,
            window
        )
        return result
    return do_prediction_proba_su(data,
                                  valid,
                                  model_dict,
                                  config_dict,
                                  meta,
                                  window)


def gather_data_for_prediction(prediction_dict, window):
    """ Gather all the data from the input rasters and return validity raster

    Args:
        prediction_dict (dict): parameters for the prediction
        window (rasterio Window): window of the chunk to predict

    Returns:
        data (np.ndarray): Array with data for prediction
        valid (np.ndarray): Validity array
    """
    with rasterio.open(prediction_dict['rasters'][0], 'r') as first:
        f_data = first.read(window=window)
        # Set the correct sizes for output blocks and set the transformation
        for meta_type in ['dst_meta', 'dst_proba_meta']:
            prediction_dict['meta'][meta_type].update(
                width=f_data.shape[2],
                height=f_data.shape[1],
                transform=first.window_transform(window))

        # Create the data array
        data_array = np.zeros((prediction_dict['meta']['bandcount'],
                               f_data.shape[1],
                               f_data.shape[2]))

        data, valid = iterate_rasters_for_data(window,
                                               first,
                                               f_data,
                                               data_array,
                                               **prediction_dict)
        return data, valid

def gather_data_for_prediction_ts(prediction_dict, window):
    """Gathers all the data neccessary for a prediction using a raster TS

        Args:
            prediction_dict(dict): raster dates and paths
            window(rasterio.Window): window to get data for
        Returns:
            data (np.array): gathered and imputed data
            valid (np.array): Array with indices of locations of valid data
     """
    # Get all data in a df with pixels as columns and dates and bands as the
    # indices
    with rasterio.open(prediction_dict['rasters'][0], 'r') as src:
        shape = src.read(window=window).shape
        x_min, y_min, x_max, y_max = src.window_bounds(window)
        for meta_type in ['dst_meta', 'dst_proba_meta']:
            prediction_dict['meta'][meta_type].update(
                width=shape[2],
                height=shape[1],
                transform=src.window_transform(window))

    window_polygon = [[(x_min, y_min),
                       (x_min, y_max),
                       (x_max, y_max),
                       (x_max, y_min)]]

    rois = {'properties': {'id': 1},
            'geometry': {'coordinates': window_polygon,
                         'type':'Polygon'}}

    pixel_data = get_timeseries_samples(prediction_dict['rasters'],
                                        [rois],
                                        out_dir=None,
                                        config_dict=prediction_dict[
                                            'config_dict'])
    data = pixel_data.transpose().values
    # All TS is imputed, so we assume that all data is valid.
    valid = np.ones(shape[1:], dtype=np.bool)
    return data, valid


def prediction(window, prediction_dict):
    """ Do the prediction for a separate window and save to a tmp file

    Args:
        window A rasterio window.Window
        prediction_dict: dictionary of arguments:

            rasters  The input raster list
            model_dict Dictionary specifying the algorithm, model labels
            meta The meta files from source, destination and dest_proba
            config_dict Command line arguments
            """
    if prediction_dict['config_dict']['app_rasters_are_timeseries']:
        data, valid = gather_data_for_prediction_ts(prediction_dict, window)
    else:
        data, valid = gather_data_for_prediction(prediction_dict, window)
    if not valid.any():
        return

    if not prediction_dict['config_dict']['app_algorithm'] in \
           US_ALGORITHMS:
        result = do_prediction_proba(
            data,
            valid,
            prediction_dict,
            window)

    else:
        labels_external = prediction_dict['model_dict']['model'].predict(data)
        result = np.full(valid.shape, -9999, np.int32)
        result[valid] = labels_external

    # Write the classification results to tifs
    write_tifs(os.path.join(TMP_DIR, 'classification'),
               window,
               prediction_dict['meta']['dst_meta'],
               result)


def predict_proba_single_class(data,
                               valid,
                               model_dict,
                               result,
                               meta,
                               window):
    """
    Predict the probability for an array
    Args:
        data: input data (array)
        valid: where input data is valid (array)
        model_dict: Model dictionary containing algorithm info
        result: (empty) result array
        meta: rasterio output meta dictionary
        window: rasterio window

    Returns:
        predicted probability (array
        labels (list)
        meta: rasterio meta for output
    """
    meta.update(dtype=rasterio.dtypes.float32, nodata=-9999)
    result = np.full(valid.shape, -9999, np.float32)
    labels_internal_highest_p = model_dict['model'].decision_function(data)
    result_proba = result.copy().astype(np.float32)
    result_proba[valid] = labels_internal_highest_p

    write_tifs(os.path.join(TMP_DIR, 'probability'),
               window,
               meta,
               result_proba)


def predict_single_class(data,
                         valid,
                         model_dict,
                         config_dict,
                         meta,
                         window):
    """
    Predict the probability for an array
    Args:
        data: input data (array)
        valid: where input data is valid (array)
        model_dict: Model dictionary containing algorithm info
        config_dict: Configuration dictionary
        meta: rasterio output meta dictionary
        window: rasterio window

    Returns:
        predicted array

    """
    # Get the final classification of the highest probabilities
    predicted = model_dict['model'].predict(data)
    # print(np.unique(predicted))
    result = np.full(valid.shape, -9999, np.int32)

    # See if we need to output some of the modelled probabilities
    if config_dict['su_probability']:
        predict_proba_single_class(data,
                                   valid,
                                   model_dict,
                                   result,
                                   meta,
                                   window)
    result[valid] = predicted
    return result
