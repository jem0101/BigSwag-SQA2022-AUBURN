"Training module for classifier"
import numpy as np

from sklearn.model_selection import train_test_split

from classifier.utils.general import get_available_model_args
from classifier import __version__ as classifier_version
from classifier.settings import ALGORITHM_DICT


def train_dataset(dataset, algorithm, algorithm_args, out_dir,
                  config_dict):
    """Train the model using a dataset

    Args:
        dataset         Dataset containing features in the columns wiht one
                        column named "class" which contains class labels or
                        \numbers
        algorithm       The algorithm name
        algorithm_args  The algorthm options (dict)
        out_dir         Path where to write dataset file
        config_dict     command line arguments dictionary

    Returns:
        model_dict      A dictionary containing the name, model and label
                        encoder.
        test            A test dataset which was not used during training
    """
    # Encode the labels
    labels = np.unique(dataset['class'].values).tolist()
    # Split the dataset,
    model, xcols, test = init_model_and_train(dataset,
                                              algorithm,
                                              algorithm_args,
                                              out_dir,
                                              config_dict)

    model_dict = {'app_algorithm': algorithm,
                  'model': model,
                  'labels': labels,
                  'names': xcols,
                  'version': classifier_version}
    return model_dict, test

def set_model_parameters(algorithm, algorithm_args, y_train):
    """
    Set the model parameters
    Args:
        algorithm:  Algorithm name
        algorithm_args: Algorithm arguments
        y_train: The training dataset classes

    Returns:
        parametrized model

    """
    model_type = ALGORITHM_DICT[algorithm]
    model = model_type()
    model_algorithm_args = get_available_model_args(algorithm_args, model_type)
    model.set_params(**model_algorithm_args)
    n_class = len(set(y_train))
    if algorithm == 'xgboost' and n_class < 3:
        model.set_params(**{'objective': 'binary:logistic'})
    return model

def init_model_and_train(dataset, algorithm, algorithm_args, out_dir,
                         config_dict):
    """
    Set the model parameters and train it
    Args:
        dataset (Array) : The dataset for input in the model (array)
        algorithm: The algorithm to use
        algorithm_args: The parameters for the algorithm
        out_dir: The output directory

    Returns:
        Trained model
        Names of bands
        test dataset

    """
    optimize = config_dict['su_optimize']
    optimize_iters = config_dict['su_optimize_number']
    test_size = config_dict['acc_testfraction']
    train, test = train_test_split(dataset, test_size=test_size)
    xcols = [x for x in train.columns if 'class' not in x]
    x_train = train[xcols].values
    y_train = np.ravel(train[[x for x in train.columns if 'class' in x]].values)
    # Get the model and fit
    model = set_model_parameters(algorithm, algorithm_args, y_train)
    if optimize:
        try:
            model.parameter_matrix = {
                'max_features': config_dict['su_search_max_features'],
                'max_depth': config_dict['su_search_max_depth'],
                'max_leaf_nodes': config_dict['su_search_max_leaf_nodes'],
                'n_estimators': config_dict['su_search_n_estimators']}
            algorithm_args = model.random_optimise(
                x_train,
                y_train,
                out_dir,
                optimize_iters
                )
            model = set_model_parameters(algorithm, algorithm_args, y_train)
        except AttributeError:  # if there is no parameter matrix
            raise NotImplementedError
    # y_train_encoded = sample_labels_encoder.transform(y_train)
    model.fit(x_train, y_train)
    return model, xcols, test
