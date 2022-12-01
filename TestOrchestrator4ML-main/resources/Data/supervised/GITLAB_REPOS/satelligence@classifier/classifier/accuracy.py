"Accuracy assessment and plotting for classifier"
import itertools
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import fiona
import rasterio
from rasterio import mask
from shapely.geometry import shape
from shapely.ops import cascaded_union


ACCURACY_LOGGER = logging.getLogger(__name__)


def write_confusion_matrix(model_dict, test_set, cm_fn, plot=True, csv=True):
    """ Uses a sklearn confusion matrix (np 2d array) to write to a csv file
    that also contains overall metrics. Also plots the confusion matrix for
    easier viewing.

        Args:
            model_dict (dict):dictionary with the name, model and label encoder
            test_set (pd.DataFrame): test dataset not used during training
            cm_fn (str): The filename of the output csv
            plot (str): Whether or not to plot the figure
            csv (str): Whether or not to write csv file

        returns:
            nothing
    """
    # Convert dict and dataframe to arrays
    x_test = test_set[[x for x in test_set.columns if 'class' not in x]].values
    y_test = np.ravel(
        test_set[[x for x in test_set.columns if 'class' in x]].values)
    preds = model_dict['model'].predict(x_test)
    compute_confusion_matrix(y_test, preds, cm_fn, plot, csv)


def compute_confusion_matrix(y_test, preds, cm_fn, plot=True, csv=True):
    """ Uses a sklearn confusion matrix (np 2d array) to write to a csv file
    that also contains overall metrics. Also plots the confusion matrix for
    easier viewing.

        Args:
            y_test (np.array): Ground truth (correct) target values
            preds (np.array): Estimated targets as returned by a classifier
            cm_fn (str): The filename of the output
            plot (str): Whether or not to plot the figure
            csv (str):  Whether or not to write the csv
    """

    # Write and plot confusion Matrix
    cm_labels = sorted(list(set(np.unique(y_test)) | set(np.unique(
        preds))))
    kappa = cohen_kappa_score(y_test, preds, labels=cm_labels)
    conf_matrix = confusion_matrix(y_test, preds, labels=cm_labels)

    # Metrics: overall mean accuracy and kappa
    ACCURACY_LOGGER.info("\n####-----Accuracy Assessment-----#####\n")
    metrics_list = []
    metrics_list.append('Overall Accuracy {:.4f}'.format(
        np.trace(conf_matrix)/np.nansum(conf_matrix)))
    metrics_list.append('Kappa {:.4f}'.format(kappa))
    for metric in metrics_list:
        ACCURACY_LOGGER.info(metric)

    if csv:
        df_cm = pd.DataFrame(
            data=conf_matrix, index=cm_labels, columns=cm_labels)
        # Total of the rows
        df_cm['Total'] = df_cm.sum(axis=1)
        # Accuracy
        df_cm['Accuracy'] = np.diag(df_cm[cm_labels]) / df_cm['Total']
        # Appending the numbers to the df
        df_cm = df_cm.append(df_cm.sum(axis=0).rename('Total')).append(
            (pd.Series(np.diag(conf_matrix) / np.sum(conf_matrix, axis=0),
                       index=cm_labels).rename('Reliability')))
        df_cm.to_csv(cm_fn+'.csv', sep=';', float_format='%.4f')

    if plot:
        # Plotting if necessary
        try:
            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(conf_matrix,
                                  classes=cm_labels,
                                  normalize=True
                                 )
            plt.savefig(cm_fn + '_plot.png', dpi=150)
        except AttributeError:  # Not all models have FIs, so skip
            pass


def plot_confusion_matrix(conf_matrix, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          labeldict=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') /\
                      conf_matrix.sum(axis=1)[:, np.newaxis]
    if labeldict is not None:
        classes = map(labeldict.get, classes)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]),
                                  range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black",
                 fontsize=4)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_feature_importances(model_dict, outfile):
    """ Plot the feature importances of the forest"""
    importances = model_dict['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    labels = [model_dict['names'][x] for x in indices]
    nr_features = len(model_dict['names'])
    fig, axis = plt.subplots()
    axis.set_title("Feature importances (band numbers)")
    # Get STD whenever possible
    if model_dict['app_algorithm'] == 'randomforest':
        std = np.std([tree.feature_importances_ for tree in model_dict[
            'model'].estimators_], axis=0)
        axis.bar(range(nr_features), importances[indices],
                 color="r", yerr=std[indices], align="center")
    else:
        axis.bar(range(nr_features), importances[indices],
                 color="r", align="center")
    axis.set_xticks(range(nr_features))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_xlim([-1, nr_features])
    fig.tight_layout()
    plt.savefig(outfile, dpi=300)


def __gather_samples_for_roi(roi, rasterfile):
    """Get sample values from rasters

        Args:
            roi (fiona shape): region of interest
            raster (str): raster file name
            warp_dst (dict): warp specification

        Returns:
            np.array [nsamples, nfeatures] of samples from within rois.

    """
    with rasterio.open(rasterfile) as src:
        selection, _, window = mask.raster_geometry_mask(
            src,
            [roi["geometry"]],
            crop=True)
        raster_subset = src.read(window=window)

        # rasterio convention: outside shape=True, inside=False. We invert.
        samples_per_raster = raster_subset[:, ~selection]
        roi_values = np.asarray(samples_per_raster).flatten()

    return roi_values


def assess_accuracy(raster, rois, cm_fn, subset=None):
    """Accuracy assessment
        Args:
            raster (str): filename of classication raster
            rois (str): rois filename
            cm_fn (str): filename of output
            subset (str): filename of subset used for classication
    """

    start = time.time()

    # Is subset is given, exclude its polygons from accuracy assessment
    if subset:
        polygons = [
            shape(feature['geometry']) for feature in fiona.open(subset)]
        union_subset = cascaded_union(polygons)

    sample_labels = []
    all_samples = []
    counter = 0
    with fiona.open(rois, "r") as shapefile:
        for roi in shapefile:
            if subset and shape(roi['geometry']).intersects(union_subset):
                counter += 1
                continue
            roi_samples = __gather_samples_for_roi(
                roi,
                raster
            )
            if roi_samples is not None:
                all_samples += list(roi_samples.astype(np.float))
                roi_id = int(roi['properties']['id'])
                sample_labels += len(roi_samples) * [roi_id]
        all_samples = np.asarray(all_samples)
        sample_labels = np.asarray(sample_labels)

        ACCURACY_LOGGER.info(
            "Using %i groundtruth polygons, \
                excluding %i already used in classifier",
            len(shapefile),
            counter)

    #write confusion matrix
    compute_confusion_matrix(
        sample_labels, all_samples, cm_fn, plot=True, csv=False)

    ACCURACY_LOGGER.info(
        "Total run time was %i seconds", (int(time.time() - start)))


if __name__ == "__main__":
    START = time.time()
    RASTER = "classification.tif"
    ROIS = "merged_gewassen.geojson"
    SUBSET = "subset_merged_gewassen.geojson"
    CM_FN = "confusion_matrix_total"

    assess_accuracy(RASTER, ROIS, CM_FN, SUBSET)

    print(
        "Total run time was {} seconds".format(int(time.time() - START)))
