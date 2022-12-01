"""All the necessary functions for preparing the data for the classification
model can be found here.."""
import base64
import logging
import os
import sys

import fiona
import folium
from folium import IFrame
import geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape, MultiPoint
from sklearn.ensemble import IsolationForest

from classifier.utils.general import progress, impute_values
from classifier.utils.raster import raster_warp_dst

# Set Logging levels on Rasterio and Fiona to not get too many logging msg
RASTERIO_LOGGER = logging.getLogger('rasterio')
RASTERIO_LOGGER.setLevel(logging.CRITICAL)
FIONA_LOGGER = logging.getLogger('fiona')
FIONA_LOGGER.setLevel(logging.CRITICAL)
DATAPREP_LOGGER = logging.getLogger(__name__)


def check_training_data_nodata(dataset, config_dict):
    """
    Check whether there is nodata in the training dataset
    Args:
        dataset (DataFrame or Numpy Array):Dataset with gathered sample data
        config_dict (dict): CLI configuration dictionary

    Returns:
        dataset without nans

    """
    #Check if nans are present
    contains_nans = False
    if isinstance(dataset, np.ndarray):
        if np.isnan(dataset).any():
            contains_nans = True
    else:
        if dataset.isnull().values.any():
            contains_nans = True

    if contains_nans and not config_dict['app_imputation']:
        DATAPREP_LOGGER.warning("""The sampled dataset contains nan's.
        Imputation is set to False so all rows containing NaNs will be 
        deleted. If you want to prevent this from happening, please set the 
        app_imputation parameter to True and choose a strategy and constant 
        if necessary.""")
        if isinstance(dataset, np.ndarray):
            dataset = dataset[~np.isnan(dataset).any(axis=1)]
        else:
            dataset = dataset.dropna(how='any')
    elif contains_nans:
        DATAPREP_LOGGER.info("Found missing values. Doing imputation...")
        if config_dict['app_imputation_strategy'] == 'randomforest':
            DATAPREP_LOGGER.error("Random Forest strategy only allowed for "
                                  "timeseries, please select another strategy")
            sys.exit()
        dataset = impute_values(dataset, config_dict)
    return dataset


def check_training_extent():
    """Check whether the extent of the rois falls outside of the extent of the
        rasters"""
    raise NotImplementedError


def get_bandnames(rasters):
    """
    Get the filenames and add a number for the bands
    Args:
        rasters: The list of input rasters

    Returns:
            List of rasternames and bandsuffixes
    """
    bandnames = []
    for files in rasters:
        rastername = files.split('/')[-1]
        with rasterio.open(files) as src:
            nr_bands = src.count
            bandnames.extend(['{}_{}'.format(rastername, x) for x in
                              np.arange(1, nr_bands+1)])
    return bandnames


def gather_samples_for_roi(roi, rasters, warp_dst):
    """Get sample values from rasters warped to the specified warp destination

        Args:
            roi (fiona shape): region of interest
            rasters (list): list of raster file names
            warp_dst (dict): warp specification

        Returns:
            np.array [nsamples, nfeatures] of samples from within rois.

    """

    try:
        roi_values = []
        for files in rasters:
            with rasterio.open(files) as src:
                with WarpedVRT(
                    src,
                    resampling=Resampling.nearest,
                    width=warp_dst["width"],
                    height=warp_dst["height"],
                    transform=warp_dst["transform"],
                    crs=warp_dst["crs"]
                ) as vrt:
                    selection, _, window = mask.raster_geometry_mask(
                        vrt,
                        [roi["geometry"]],
                        crop=True)
                    raster_subset = vrt.read(window=window)

            # rasterio convention: outside shape=True, inside=False. We invert.
            samples_per_raster = raster_subset[:, ~selection]
            roi_values.append(samples_per_raster)
        roi_values = np.concatenate(roi_values, axis=0).T
    except ValueError:
        DATAPREP_LOGGER.debug("ROI %s OUT OF BOUNDS.. Continuing without "
                              "it", roi['geometry']['coordinates'][0][0])
        roi_values = None
    return roi_values


def gather_samples(rasters, rois, bandnames, out_dir, config_dict):
    """Takes a raster reference from the first raster, loops over all rois and
       returns two lists of equal lenght for samples and labels.s

        Args:
            rasters (list): list of raster file names
            rois (str): path to fiona supported regions of interest file
            bandnames (list): list of names of the input rasters and bands
            out_dir (str): path to the output directory
            config_dict (dict): The cli arguments
        Returns:
            Dataset (DataFrame): Gathered Samples

    """
    sample_labels = []
    all_samples = []
    roi_dict = {}
    if config_dict['su_boxplots']:
        #Boxplot Directory
        roi_bp_dir = out_dir + '/roi_boxplots/'
        os.mkdir(roi_bp_dir)

    # reference everything to first raster
    warp_dst = raster_warp_dst(rasters[0])

    with fiona.open(rois, "r") as shapefile:
        nr_regions = len(shapefile)
        counter = 0
        DATAPREP_LOGGER.debug("%s contains %i regions", rois, nr_regions)
        sample_count = 0
        DATAPREP_LOGGER.info("--Getting Samples--")
        for roi in shapefile:
            progress(counter, nr_regions)
            roi_samples = gather_samples_for_roi(
                roi,
                rasters,
                warp_dst
            )
            if roi_samples is not None:
                sample_count += roi_samples.shape[0]

                all_samples.append(roi_samples.astype(np.float))
                roi_id = int(roi['properties']['id'])
                sample_labels += len(roi_samples) * [roi_id]
                if config_dict['su_boxplots']:
                    polygon_boxplot(roi_samples,
                                    bandnames,
                                    roi_id,
                                    roi_bp_dir,
                                    counter)
                roi_dict[counter] = shape(roi['geometry'])
            counter += 1
        all_samples = np.concatenate(all_samples, axis=0)
        sample_labels = np.asarray(sample_labels)

    all_samples = check_training_data_nodata(all_samples, config_dict)

    dataset = pd.DataFrame(all_samples, columns=bandnames).join(
        pd.DataFrame({'class': sample_labels}))
    DATAPREP_LOGGER.debug("Got: %i samples in total.", len(dataset))

    if config_dict['su_remove_outliers']:
        dataset = outlier_removal(dataset)
    if config_dict['su_boxplots']:
        boxplot(dataset, out_dir)
        folium_map_boxplots(roi_dict, roi_bp_dir)
    return dataset


def createdataset(rasters, rois, workspace, config_dict):
    """
    Create the entire dataset based on rasters and rois
    Args:
        rasters (list): the input rasters
        rois (str): The ogr vector file containing polygons
        workspace (str): Path to workspace
        config_dict (dict): The cli arguments
    Returns:
        dataset
    """
    # Gather the samples and return a full df of all samples
    bandnames = get_bandnames(rasters)
    return gather_samples(rasters,
                          rois,
                          bandnames,
                          workspace,
                          config_dict)

def polygon_boxplot(roi_values, bandnames, class_title, roi_bp_dir, roi_number):
    """Make a simple boxplot for a polygon from the training data

        Args:
            roi_values (np array): A flattened array of roi values for all bands
            bandnames (list): The names of the input bands
            class_title (str): The (landuse) class of the roi
            roi_bp_dir (str): Path where the boxplot figure is saved
            roi_number (int/str): The identifier of the roi to use in filename

    """
    #Bunch of settings to make the boxplots look nicer
    plt.rcParams['xtick.labelsize'] = 4
    plt.rcParams['ytick.labelsize'] = 4
    plt.rcParams['axes.labelsize'] = 4
    plt.rcParams['font.size'] = 4
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['lines.linewidth'] = 0.2
    plt.rcParams['boxplot.boxprops.linewidth'] = 0.2
    plt.rcParams['boxplot.capprops.linewidth'] = 0.2
    plt.rcParams['boxplot.flierprops.linewidth'] = 0.2
    plt.rcParams['boxplot.flierprops.markersize'] = 2
    plt.rcParams['boxplot.meanprops.linewidth'] = 0.2
    plt.rcParams['boxplot.meanprops.markersize'] = 2
    plt.rcParams['boxplot.medianprops.linewidth'] = 0.2
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 0.2

    sample_df = pd.DataFrame(roi_values, columns=bandnames)
    n_plots = len(sample_df.columns)
    fig, axarr = plt.subplots(1, n_plots)
    for i, columns in enumerate(sample_df.columns):
        subset = sample_df[columns]
        subset.plot(kind='box', ax=axarr[i])
    fig.suptitle(class_title)
    fig.set_size_inches(5, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(roi_bp_dir, "{}.png".format(roi_number)),
                dpi=300)
    plt.close()

def boxplot(samples, output_dir):
    """Boxplot of all samples

        Args:
            samples (Dataframe): All sample dataset
            output_dir (str): The directory where to save the boxplot

    """
    classes = set(samples['class'])
    n_plots = len(classes)
    if n_plots > 1:
        _, axarr = plt.subplots(n_plots)
        for i, class_number in enumerate(classes):
            subset = samples[samples['class'] == class_number]\
                .drop('class', axis=1)
            subset.boxplot(ax=axarr[i])
            axarr[i].set_title(class_number)
    else:
        _, axis = plt.subplots()
        cols = [x for x in samples.columns if not x == 'class']
        samples[cols].boxplot(ax=axis)
    plt.tight_layout()
    plt.savefig(output_dir+'/all_samples_boxplot.png')


def add_polygons_to_map(gdf, folium_map, png_dir):
    """Adds the separate rois to the map and creates popups with boxplots

        Args:
            gdf (GeoDataFrame): Contains all polygons to add
            folium_map (folium map instance): map to where polygons are added
            png_dir (str): Location of the pngs

        Returns:
            folium map with polygons added

    """
    for items in gdf.iterrows():
        geom_geojson = geojson.Feature(geometry=items[1]['geometry'],
                                       properties={}
                                      )
        points = geom_geojson['geometry']['coordinates']
        image_link = '{}/{}.png'.format(png_dir, items[1]['id'])
        encoded = base64.b64encode(open(image_link, 'rb').read()).decode()
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = IFrame(html(encoded), width=1600, height=900)
        popup = folium.Popup(iframe, max_width=2560)
        switched_points = [[y, x] for x, y in points[0]]
        marker = folium.PolyLine(switched_points,
                                 radius=10,
                                 popup=popup,
                                 color='#3186cc',
                                 fill=True,
                                 fill_color='#3186cc'
                                )
        folium_map.add_child(marker)
    return folium_map


def add_baselayers(folium_map):
    """Adds the baselayers to the folium map

        Args:
            folium_map(Map): Map to add the baselayers

        Returns:
            folium_map(Map): Map with baselayers

    """
    folium.TileLayer(name='Google Satellite',
                     tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                     attr='Google Imagery'
                    ).add_to(folium_map)
    folium.TileLayer(name='Esri Satellite',
                     tiles='https://server.arcgisonline.com/ArcGIS/rest'
                           '/services/World_Imagery/MapServer/tile/{z}/{y}/{'
                           'x}',
                     attr='Esri Imagery').add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    return folium_map

def folium_map_boxplots(roi_dict, out_dir):
    """Main function to make folium map with boxplots

        Args:
            roi_dict (dict): Roi names and geometries
            out_dir (str): Output directory

    """
    indices = list(roi_dict.keys())
    geoms = list(roi_dict.values())
    gdf = gpd.GeoDataFrame({'id': indices, 'geometry':geoms})
    center_point = MultiPoint(gdf.centroid).centroid
    lat, lon = center_point.y, center_point.x
    folium_map = folium.Map(location=[lat, lon],
                            zoom_start=9,
                           )
    folium_map = add_baselayers(folium_map)
    folium_map = add_polygons_to_map(gdf, folium_map, out_dir)
    folium_map.save(out_dir+'/all_rois.html')

def outlier_removal(samples):
    """Outlier removal from samples using Isolation Forest

        Args:
            samples (DataFrame)

        returns:
            samples (DataFrame)
    """
    DATAPREP_LOGGER.info("Now checking for outliers")

    cols = [x for x in samples.columns if x != 'class']
    if_model = IsolationForest(contamination='auto', behaviour='new')

    samples['inlier'] = 0
    lulc_classes = samples['class'].unique()

    # Temporarily here because of annoying warnings in scikit-learn module
    # Needs to be removed after updating scikit-learn to 0.22.0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        for lulc_class in lulc_classes:
            samples.loc[samples['class'] == lulc_class, 'inlier'] = \
                if_model.fit_predict(samples[samples['class'] == lulc_class][
                    cols])

    # get the number of removed pixels and give output
    class_counts(samples)
    cols_to_return = [x for x in samples.columns if not x == 'inlier']

    return samples[samples['inlier'] > 0][cols_to_return]



def class_counts(samples):
    """Prints the number of pixels removed after outlier removal

        Args:
            samples (DataFrame): Gathered samples with 'inlier' column

    """
    classes = samples['class'].unique()
    count_dict = {}
    for lulc_class in classes:
        count_dict[lulc_class] = [
            samples[samples['class'] == lulc_class].count()[0],
            samples[(samples['class'] == lulc_class) &
                    (samples['inlier'] < 1)].count()[0]
        ]

    to_print = "\t\t{:<8} \t{:<15} \t{:<15} \t{:<15}".format(
        "Class",
        "#Original Pixels",
        "Pixels Removed",
        "Remaining fraction")
    DATAPREP_LOGGER.info("%s", to_print)

    for class_label, pixel_count in list(count_dict.items()):
        original, removed = pixel_count
        to_print = "\t\t{:<8} \t{:<15} \t{:<15} \t{:<15.2f}".format(
            class_label,
            original,
            removed,
            (1 - removed/original))
        DATAPREP_LOGGER.info("%s", to_print)
