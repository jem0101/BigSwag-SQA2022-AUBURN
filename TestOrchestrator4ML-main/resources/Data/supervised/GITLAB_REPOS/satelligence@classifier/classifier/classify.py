"""An sk-learn classifier for landcover classification"""

import logging
import os
import time
import sys

import fiona
import pandas as pd

from classifier.dataprep import createdataset, outlier_removal
from classifier.train import train_dataset
from classifier.accuracy import write_confusion_matrix, plot_feature_importances
from classifier.predict import prediction
from classifier.unsupervised import make_input_array, train_kmeans
from classifier.utils.general import multiprocess_function, progress, \
    save_model, read_model
from classifier.utils.raster import stitch, get_meta
from classifier.timeseries import get_timeseries_samples
from classifier.settings import WORKSPACE, US_ALGORITHMS

START = time.time()
CLASSIFIER_LOGGER = logging.getLogger(__name__)


def read_samples(samples, remove_outliers=True):
    """Read a samples file.

    Args:
        samples(path): CSV file containing samples
        remove_outliers (Bool): Flag for removing outliers from samples

    Returns:
        sample_df(DataFrame): DF containing samples

    """
    sample_df = pd.read_csv(samples, index_col=0)
    if remove_outliers:
        sample_df = outlier_removal(sample_df)
    return sample_df

def get_rois_extent(rois):
    """

    Args:
        rois (str): path to a rois file

    Returns:
        bounds of rois (list): ulx, uly, llx, lly coordinates

    """
    with fiona.open(rois, "r") as shapefile:
        return shapefile.bounds

def gather_samples(rois, rasters, out_dir, config_dict):
    """Gather samples from a rois file and raster files

    Args:
        rois: The path to a rois file
        rasters: A list of raster files
        out_dir: Path where to write the dataset file
        config_dict: The CLI arguments dictionary

    Returns:
        A dataframe containing pixel values and classes

    """
    # Train a model and continue
    samples = createdataset(
        rasters,
        os.path.join(WORKSPACE, rois),
        out_dir,
        config_dict)
    samples.to_csv(os.path.join(out_dir, 'samples.csv'))
    return samples


def train(samples, rasters, out_dir, config_dict, rois_extent):
    """Train a model using samples and rasters

    Args:
        samples: Samples dataframe containing pixel values and class values
        rasters: A list of raster files
        out_dir: Path where to write the dataset file
        config_dict:  The command line arguments dictionary
        rois_extent: The extent of the training data (if any, else None)

    Returns:
        The return value. True for success, False otherwise.

    """
    CLASSIFIER_LOGGER.info("\n####-----Training----#####\n")
    dataset = samples
    windows, _ = get_meta(rasters, config_dict['app_window'])
    algorithm_args = {'n_jobs': config_dict['app_threads']}
    if config_dict['app_algorithm'] in US_ALGORITHMS and len(samples) < 1:
        # Do unsupervised
        array = make_input_array(
            rasters,
            windows,
            config_dict
        )
        model_dict = train_kmeans(array,
                                  config_dict,
                                  algorithm_args)
    else:  # All supervised methods
        model_dict, test = train_dataset(
            dataset,
            config_dict['app_algorithm'],
            algorithm_args,
            out_dir,
            config_dict
            )
        if config_dict['app_algorithm'] in ['randomforest', 'xgboost'] \
                and config_dict['acc_perform_assesment']:
            # Do the accuracy analysis
            cm_fn = os.path.join(out_dir, 'confusion_matrix')
            fi_fn = os.path.join(out_dir, 'feature_importance.png')
            write_confusion_matrix(model_dict, test, cm_fn)
            plot_feature_importances(model_dict, fi_fn)
    # save the model as a python pickle
    if not rois_extent is None:
        model_dict['rois_bounds'] = rois_extent
    if config_dict['model_save']:
        save_model(model_dict, out_dir, config_dict)
    CLASSIFIER_LOGGER.info("\nFinished Training\n")
    return model_dict


def moving_predict(args):
    """prediction function per thread for a window

    Args:
        args: A list containing the following arguments
                window A rasterio window
                rasters The raster files
                model model dictionary
                meta    meta of raster dict
                args    CLI arguments dict
                block_count the total number of blocks int

    Returns:
        Nothing

    """
    window_nr = args[0][0]
    win = args[0][1]
    arg_names = ['rasters', 'model_dict', 'meta', 'config_dict']
    prediction_dict = dict(zip(arg_names, args[1:5]))
    prediction(win, prediction_dict)
    progress(window_nr, int(args[5]))
    return None


def predict(model_dict, rasters, out_dir, config_dict):
    """Prediction function using a trained model and raster files

    Args:
        model_dict: A trained model
        rasters: A list of raster files
        out_dir: Path where to write the dataset file
        config_dict:  The command line arguments dictionary

    """

    CLASSIFIER_LOGGER.info("\n####-----Prediction----#####\n")
    windows, meta = get_meta(rasters, config_dict['app_window'])
    block_count = len(windows)

    iterable = [[x,
                 rasters,
                 model_dict,
                 meta,
                 config_dict,
                 block_count]
                for x in windows]
    if config_dict['app_threads'] > 1 or config_dict['app_threads'] == -1:
        if model_dict['app_algorithm'] == 'xgboost':
            # # XGBoost somehow only works with 1 thread if you set both to 1
            raise Exception(
                "\nXGBoost does not support n_threads different than 1.\n"
                "Please change the app_threads to 1 in the config file")
        if not model_dict['app_algorithm'] in US_ALGORITHMS:
            model_dict['model'].set_params(**{'n_jobs': 1})
        multiprocess_function(moving_predict,
                              iterable,
                              config_dict['app_threads'])
    else:
        for wins in iterable:
            moving_predict(wins)

    # ##--------------------STITCHING-----------------------------------###

    CLASSIFIER_LOGGER.info("\n####-----Stitching----#####\n")

    # Run the gdalwarp command in the command line
    stitch(out_dir, meta)
    CLASSIFIER_LOGGER.info(
        "Total run time was %i seconds", (int(time.time() - START)))


def train_and_predict_with_samples(samples, rasters, out_dir, config_dict,
                                   rois_extent=None):
    """Train model using samples

    Args:
        samples: Samples dataframe containing pixel values and class values
        rasters: A list of raster files
        out_dir: Path where to write the dataset file
        config_dict:  The command line arguments dictionary

    Returns:
        The return value. True for success, False otherwise.

    """
    model_dict = train(samples, rasters, out_dir, config_dict, rois_extent)
    predict(model_dict, rasters, out_dir, config_dict)


def gather_train_predict_rois(rois, rasters, out_dir, config_dict):
    """Gather samples from rois file and rasters and continue with training and
    prediction

    Args:
        rois: The path to a rois file
        rasters: A list of raster files
        out_dir: Path where to write the dataset file
        config_dict:  The command line arguments dictionary

    Returns:
        The return value. True for success, False otherwise.

    """
    samples = gather_samples(rois, rasters, out_dir, config_dict)
    rois_extent = get_rois_extent(rois)
    return train_and_predict_with_samples(samples,
                                          rasters,
                                          out_dir,
                                          config_dict,
                                          rois_extent)

def classify_(rasters, config_dict, out_dir, rois=None):
    """Entry point function for pixel-based classification

        Args:
            rasters (list): List of raster paths
            config_dict (dict): CLI configuration dict
            out_dir (str): Path to output directory
            rois (str): Path to rois file

    """
    if not config_dict['app_model'] is None:
        model = read_model(config_dict['app_model'])
        predict(model, rasters, out_dir, config_dict)
    elif not config_dict['app_samples'] is None:
        samples = read_samples(
            config_dict['app_samples'],
            config_dict['su_remove_outliers'])

        train_and_predict_with_samples(samples,
                                       rasters,
                                       out_dir,
                                       config_dict)

    elif rois is not None:
        gather_train_predict_rois(
            rois,
            rasters,
            out_dir,
            config_dict)
    else:
        CLASSIFIER_LOGGER.info("No models, samples or rois provided. Doing "
                               "unsupervised classification")
        if not config_dict['app_algorithm'] in US_ALGORITHMS:
            config_dict['app_algorithm'] = 'us_kmeans'
        train_and_predict_with_samples([],
                                       rasters,
                                       out_dir,
                                       config_dict)
def classify_timeseries(rasters: list,
                        config_dict: dict,
                        out_dir: str,
                        rois: str):
    """Main function to classifiy timeseries.

    For now, only from start to finish is supported. ie, supplying a model
    or samples does not work.

        Args:
            rasters (list): List of raster paths
            config_dict (dict): CLI configuration dict
            out_dir (str): Path to output directory
            rois (str): Path to rois file

    """
    # Gather samples and do imputation.
    CLASSIFIER_LOGGER.info("Now Getting Samples and doing imputation")
    samples_df = get_timeseries_samples(rasters, rois, out_dir, config_dict)
    # Reshape it
    samples = samples_df.T
    samples['class'] = samples.index
    if config_dict['app_algorithm'] == 'unsupervised':
        CLASSIFIER_LOGGER.error("Unsupervised classification not supported yet")
        sys.exit()
    model = train(samples,
                  rasters=rasters,
                  out_dir=out_dir,
                  config_dict=config_dict,
                  rois_extent=None)

    predict(model, rasters, out_dir, config_dict)

if __name__ == "__main__":
    pass
