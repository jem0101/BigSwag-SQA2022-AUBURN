"""General utilities"""
import concurrent.futures
import inspect
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import rasterio
from sklearn.impute import SimpleImputer

from classifier.settings import WORKSPACE, PARAMETERS, RASTER_EXTENSIONS

UTILS_GENERAL_LOGGER = logging.getLogger(__name__)

def get_available_model_args(model_args, function):
    """Gets the available arguments for the specific function
        Args:
            model_args(dict) : arguments to check
            function(func): function to check the kwargs from

        Returns:
            model_algorithm_args(list): model arguments that belong to function

    """
    kwarglist = inspect.getfullargspec(function)[0]
    model_algorithm_args = [x for x in model_args.keys() if x in kwarglist]
    return {k: model_args[k] for k in model_algorithm_args}


def progress(count, total, status=''):
    """A simple progress bar for the command line

    Args:
        count (int) the count between 0 and total
        total (int) the last iteration

    """
    count += 1
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    p_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (p_bar, percents, '%', status))
    sys.stdout.flush()



def cli_init(output_name, rasters, overwrite, config_location):
    """Initialize everything before starting real work.

    Args:
        output_name (str): name of the output directory in workspace
        raster (str): raster path(s) for input
        overwrite (bool): overwrite existing folder
        config_location (str): location of the config file


    Returns:
        Output_directory: path to output directory
        Rasters (list): list of separate raster files
        Args (dict): Dictionary of parameters to use

    """
    # Set Random Seed for reproducible results
    np.random.seed(0)

    rasters = get_raster_paths(WORKSPACE, rasters)
    config_dict = read_config(config_location)

    # Set all probabilities for output when classifiying segments
    if config_dict['segm_classify']:
        config_dict['su_all_probabilities'] = True

    output_directory = create_output_dir(WORKSPACE,
                                         output_name,
                                         overwrite)


    init_logger(output_directory, config_dict['app_log_level'])
    params_str = '\n'.join(['{} :  {}'.format(key, value) for key, value in
                            sorted(config_dict.items())])
    UTILS_GENERAL_LOGGER.info(
        "\nRunning the Classifier with the following parameters:"
        "\n  %s", params_str)
    return output_directory, rasters, config_dict



def create_output_dir(workspace, name, overwrite):
    """Create output directory.

    Args:
        workspace: The workspace Path
        name: Location of the directory
        overwrite (bool): overwrite existing folder


    Returns:
        Output directory: Path of output directory

    """
    if name is None:
        name = str(uuid4())[:6]
        UTILS_GENERAL_LOGGER.info("No name argument found. Making new folder "
                                  "called: %s", name)
    output_directory = os.path.join(workspace, name)
    if os.path.exists(output_directory):
        if overwrite:
            UTILS_GENERAL_LOGGER.warning("Overwriting existing directory!")
            shutil.rmtree(output_directory)
            os.mkdir(output_directory)
        else:
            UTILS_GENERAL_LOGGER.error(
                "Directory with name %s already exists. Either "
                "leave --name out,remove the directory %s, "
                "provide a unique name or turn on --overwrite.", name, name)
            sys.exit()
    else:
        os.mkdir(output_directory)
    return output_directory


def save_dict_as_json(out_file=None, dict_to_save=None):
    """Save default config values.

    Save default values for classifier parameters in json in current workspace.

    Args:
        out_file (str): file to write
        dict_to_save (dict): Dictionary to save

    """
    if out_file is None:
        out_file = os.path.join(WORKSPACE, 'config.json')
        dict_to_save = PARAMETERS
    with open(out_file, 'w') as config:
        config.write(json.dumps(dict_to_save,
                                sort_keys=True,
                                indent=1))


def read_config(config_file):
    """Read config parameter file and change defaults where necessary.

    Args:
        config_file (str): location of the config file
    returns:
        PARAMETERS (dict): dictionary with parameters
    """

    if os.path.isfile(config_file):
        with open(config_file) as json_config:
            config_dict = json.load(json_config)
        for key in config_dict.keys():
            PARAMETERS[key] = config_dict[key]
    else:
        UTILS_GENERAL_LOGGER.warning(
            "No local config file found. Using defaults")
    return PARAMETERS


def init_logger(output_directory, log_level):
    """Set and initialize all logging info.

    Args:
        output_directory: The location of the log file

    """
    logging.captureWarnings(True)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(output_directory, 'stdout.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # # create console handler and set level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    # # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # # add formatter to ch
    console_handler.setFormatter(formatter)
    # # add Handlers to logger
    logging.getLogger('').addHandler(console_handler)


def impute_values(dataset, args):
    """Impute values

    Uses the sklearn SimpleImputer to impute missing values.

    Args:
        dataset (DataFrame): DataFrame containing Nans
        args (dict): Parameters to use for imputation

    returns:
        dataset (DataFrame): DataFrame with Nans imputed
    """
    # remove inf if present
    if isinstance(dataset, np.ndarray):
        dataset = np.where(np.isinf(dataset), np.nan, dataset)
    else:
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
    verbosity_level = 0 if args['app_log_level'] == 'INFO' else 2
    if np.isnan(dataset).all() and not args['app_imputation_strategy'] == \
                                       'constant':
        UTILS_GENERAL_LOGGER.warning(
            "Empty window found, Falling back to constant "
            "strategy")
        strategy = 'constant'
    else:
        strategy = args['app_imputation_strategy']

    imputer = SimpleImputer(strategy=strategy,
                            fill_value=args['app_imputation_constant'],
                            verbose=verbosity_level)

    return imputer.fit_transform(dataset)


def multiprocess_function(function, iterable, ncpus=1):
    """General multiprocessing function.

    This function takes another function and an argument list and spreads the
    jobs over multiple processes determined in the ncpus argument.

        Args:
            function: A function that will be executed for every item in
                      the iterable argument.
            iterable (list, dict): A list of items to execute function for. The
                                   function will be executed len(iterable)
                                   number of times
            ncpus (int): Number of cpus to use. Use -1 for all available cpus
    """
    cpu_count = mp.cpu_count()
    cpus = cpu_count if ncpus == -1 else ncpus
    if not cpus == 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as \
                executor:
            res = executor.map(function, iterable)
        # Check if there were any errors:
        errors = [x for x in res if x is not None]
        if errors:
            UTILS_GENERAL_LOGGER.error(errors)
    else:
        for i in iterable:
            function(i)


def write_tifs(temp_dir, window, meta, result):
    """
    Writes the tifs of the individual windows
    Args:
        temp_dir: Temp dir location (str)
        window: rasterio window (tuple)
        meta: rasterio meta dictionary for output
        result: Result array

    Returns:
        Nothing

    """
    # Check output dir existence
    os.makedirs(temp_dir, exist_ok=True)
    meta['compress'] = 'deflate'
    with rasterio.open(os.path.join(
        temp_dir, "c{}_{}.tif".format(window.col_off, window.row_off)),
                       'w', **meta) as dst:
        dst.write_band(1, result)


def impute(data_array, **kwargs):
    """Impute missing values for prediction

    Args:
        data_array (Numpy Array): Array containing missing values
        config_dict (dict): dictionary with all application parameters

    returns:
        Imputed data_array

    """
    shape = data_array.shape
    reshaped_array = data_array.reshape(shape[0], (shape[1] * shape[2])).T
    imputed_array = impute_values(reshaped_array, kwargs['config_dict'])
    imp_shape = imputed_array.shape
    if not shape[0] * shape[1] * shape[2] == imp_shape[0] * imp_shape[1]:
        UTILS_GENERAL_LOGGER.warning("Not all columns could be calculated. "
                                     "Falling  back to constant strategy for "
                                     "this chunk")
        kwargs['config_dict']['app_imputation_strategy'] = 'constant'
        imputed_array = impute_values(reshaped_array, kwargs['config_dict'])
    imputed_array = imputed_array.T.reshape(shape[0], shape[1], shape[2])
    return imputed_array


def save_model(model_dict, out_dir, config_dict):
    """Save the model as a pickle  and metadata  as a json and zip

        Args:
            model_dict (dict):  Contains the model as well as metadata
            out_dir (str):      Path where to write model file and meta
            config_dict (dict): Configuration parameters

    """
    to_save = model_dict.copy()
    meta_tmp = tempfile.mktemp()
    pickle_tmp = tempfile.mktemp()

    # Take the model out of the dictionairy
    model = to_save.pop('model')

    # Save the model as pickle
    with open(pickle_tmp, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Save meta as json
    save_dict_as_json(meta_tmp, to_save)
    to_write = {
        meta_tmp: '{}_meta.json'.format(config_dict['name']),
        pickle_tmp: '{}.model'.format(config_dict['name'])
    }

   # Zip to output directory
    zipfile_path = '{}/model.zip'.format(out_dir)
    write_zipfile(to_write, zipfile_path)


def write_zipfile(files_to_write, zipfile_path):
    """Writes files to a zipfile

    Args:
        files_to_write (dict): dictionary of path, name of the files to
                               write to the zipfile (e.g.
                               {
                                  '/workspace/weirdname.json':
                                  'nice_name.json'
                               }
        zipfile_path (str): Path to where the zipfile goes
    """
    with ZipFile(zipfile_path, 'w') as model_zip:
        for files in files_to_write:
            print(files, files_to_write[files])
            model_zip.write(files, files_to_write[files])

def get_raster_paths(workspace, rasters):
    """Get full paths for all rasters.

    Args:
        workspace: The workspace Path
        rasters: list of rasters from the cli

    Returns:
        list: rasters with their full paths

    """
    paths = []

    for raster in rasters:
        path = os.path.join(workspace, raster)
        if os.path.isdir(path):
            paths += [
                os.path.join(path, r)
                for r
                in sorted(os.listdir(path))
            ]
        else:
            paths.append(path)
    return [x for x in paths if os.path.splitext(x)[-1] in RASTER_EXTENSIONS]


def unzip_model_file(model_file):
    """Unzips a model file and returns paths to metadata and model pickle

    Args:
        model_file (str): path to the model zipfile

    Returns:
        meta_file (str): path to tmp metafile
        pickle_file (str): path to tmp pickle file
    """
    tmpdir = tempfile.mkdtemp()
    with ZipFile(model_file, 'r') as zipf:
        zipf.extractall(tmpdir)

    model_files = os.listdir(tmpdir)
    meta_file = os.path.join(
        tmpdir,
        [x for x in model_files if x.endswith('.json')][0]
    )
    pickle_file = os.path.join(
        tmpdir,
        [x for x in model_files if x.endswith('.model')][0]
    )
    return meta_file, pickle_file


def read_model(model_file):
    """Function to read the model file and return a pickle

    Args:
        path: The path to the model zipfile

    Returns:
        a dictionary containing a model, and metadata

    """
    # Unzip
    meta_file, pickle_file = unzip_model_file(model_file)
    # Open the metadata and print the modeltypes
    with open(meta_file) as json_config:
        model_dict = json.load(json_config)

    params_str = '\n'.join(['{} :  {}'.format(key, value) for key, value in
                            sorted(model_dict.items())])

    with open(pickle_file, 'rb') as pfile:
        model_dict['model'] = pickle.load(pfile)

    UTILS_GENERAL_LOGGER.info(
        "\nLoaded a saved model, with the following metadata:"
        "\n  %s", params_str)

    return model_dict
