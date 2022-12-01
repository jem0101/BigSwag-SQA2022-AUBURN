"""Classifier segmentation functions"""

import logging
import os
import sys
from multiprocessing import Manager

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats as rs
from rsgislib.segmentation import segutils

from classifier.settings import TMP_DIR, US_ALGORITHMS
from classifier.utils.raster import open_single_raster, get_meta, \
    rasterize_polygons
from classifier.utils.general import multiprocess_function, progress
from classifier.utils.vector import vectorize_array, gdf_polygon_to_multipolygon

SEGMENTATION_LOGGER = logging.getLogger(__name__)


def check_single_raster(rasters):
    """Check if input is single or multiple rasters.

    Segmentation can only use a single input file. If we supply multiple
    raster files, these would then first need to be combined into a vrt.

    Args:
        rasters(list): list containing paths to input rasters

    Returns:
        raster: single raster file

    """
    if len(rasters) > 1:
        SEGMENTATION_LOGGER.error("Multiple rasters not supported yet. "
                                  "Please make a vrt of your rasters and try "
                                  "again")
        sys.exit()
    return rasters[0]


def simple_segmentation(out_dir, rasters, config_dict):
    """Simple segmentation that does not use multiprocessing.

    Uses the RSGISlib segmentation to make a raster file with clumps

    Args:
        out_dir (str): Output directory for the clumps image
        rasters (list): Input rasters
        config_dict (dict): Dictionary with parameters

    """
    input_image = check_single_raster(rasters)
    output_file = os.path.join(out_dir, 'segmentation_clumps.tif')

    # RSGISLIB uses print statements... Change to debug logger
    segutils.runShepherdSegmentation(
        input_image,
        output_file,
        tmpath=TMP_DIR,
        numClusters=config_dict['segm_numClusters'],
        minPxls=config_dict['segm_minPixels'],
        distThres=config_dict['segm_distThres'],
        processInMem=config_dict[
            'segm_processInMem'],
        gdalformat='GTiff',
        noStats=True,
        sampling=config_dict['segm_sampling'],
        kmMaxIter=config_dict['segm_MaxIter'])


def vectorize_(arglist):
    """Vectorize function to be used by the mutltiprocessing function

    Arguments are supplied as a single list because the MP function does not
    allow that.

    Args:
        arglist (list) List with function arguments
            windows (list): number of window and Rasterio Window
            total_windows (int): Total number of windows
            class_files (list): List with the probability files for
                                classification
            clump_file (str): The tif file with segments
            out_dir (str):    Output Directory
            mo_list (list):    Multiprocessing List for async appending
            algorithm(str): Which algorithm to use for classification


    """
    i, window = arglist[0]
    total_windows = arglist[1]
    class_files = arglist[2]
    clump_file = arglist[3]
    out_dir = arglist[4]
    mp_list = arglist[5]
    algorithm = arglist[6]
    window_array, window_meta = open_single_raster(clump_file, window)
    progress(i, total_windows)
    window_vector = vectorize_array(window_array.astype(np.int32),
                                    transform=window_meta['transform'])
    for classes in class_files:
        class_raster = os.path.join(out_dir, classes)
        if algorithm in US_ALGORITHMS:
            polygon_stats = rs.zonal_stats(vectors=window_vector['geometry'],
                                           raster=class_raster,
                                           stats=['majority'])

            window_vector[classes] = [x['majority'] for x in polygon_stats]
        else:
            polygon_stats = rs.zonal_stats(vectors=window_vector['geometry'],
                                           raster=class_raster,
                                           stats=['mean'])
            window_vector[classes] = [x['mean'] for x in polygon_stats]
    mp_list.append(window_vector)


def vectorize_and_classify(clump_file,
                           class_files,
                           out_dir,
                           config_dict):
    """Vectorize the segmented file and classify the segments.

    Args:
        clump_file (str): The tif file with segments
        class_files (list): List with the probability files for
                            classification
        out_dir (str):    Output Directory
        config_dict (dict): Dictionary with configuration parameters

    Returns:
        GeoDataframe with vectorized segments
    """
    windows, _ = get_meta([clump_file], config_dict['app_window'])
    total_windows = len(windows)
    main_gdf = gpd.GeoDataFrame()
    with Manager() as manager:
        mp_list = manager.list([main_gdf])
        iterable = [[x,
                     total_windows,
                     class_files,
                     clump_file,
                     out_dir,
                     mp_list,
                     config_dict['app_algorithm']] for x in windows]
        multiprocess_function(vectorize_,
                              iterable,
                              config_dict['app_threads'],
                             )

        main_gdf = pd.concat(mp_list, ignore_index=True)

    if config_dict['app_algorithm'] in US_ALGORITHMS:
        return main_gdf.dissolve(by='segment_id', aggfunc='median')
    return main_gdf.dissolve(by='segment_id', aggfunc='mean')


def classify_single_class_probs(probabilities, threshold_prob):
    """Classify a geopandas dataframe polygons based on a singleclass
    probability

        Args:
            probabilities (data frame column): probabilities from sklearn
            isolationforest
            threshold_prob (float): Threshold for determining inlier or
            outlier. Default that sk-learn uses is 0

        Returns:
            column with 0 or 1 depending on probability and threshold
    """
    if probabilities < threshold_prob:
        classification = 0
    else:
        classification = 1
    return classification


def classify_polygons(clumps_gdf, tif_files, threshold_prob):
    """Classifiy the polygons based on the highest probability.

    Args:
        clumps_gdf (GeoDataFrame): polygons with probabilities per class
        tif_files (list): Names of the tif files of the probabilities
        threshold_prob (float): threshold probability for single class
        classifications

    Returns:
        clumps_gdf (GeoDataFrame): polygons with added column 'max' with
                                   class number of highest probability

    """
    class_nrs = [os.path.splitext(x)[0] for x in tif_files]
    rename_dict = dict(zip(tif_files, class_nrs))
    clumps_gdf.rename(mapper=rename_dict, axis='columns', inplace=True)
    if len(class_nrs) > 1:
        clumps_gdf['max'] = clumps_gdf[class_nrs].idxmax(axis=1)
    else:
        clumps_gdf['max'] = clumps_gdf[class_nrs[0]].apply(
            classify_single_class_probs,
            threshold_prob=threshold_prob)
    clumps_gdf.crs = {'init': 'epsg:4326'}
    return clumps_gdf



def classify_segments(out_dir, config_dict):
    """Classify segmented tif.

    Takes the segmented tif and classifies each segment by using the
    probabilities which were output by the classification.

    Args:
        out_dir(str): Path to the output directory
        config_dict (dict): Dictionary with configuration parameters
    """

    SEGMENTATION_LOGGER.info(
        "Vectorizing the segments and getting probabilities")

    # Vectorize and get probabilities
    if not config_dict['segm_custom_segments']:
        clumps_tif = os.path.join(out_dir, 'segmentation_clumps.tif')
    else:
        clumps_tif = config_dict['segm_custom_segments']

    if config_dict['app_algorithm'] in US_ALGORITHMS:
        tif_files = [os.path.join(out_dir, 'classification.tif')]
    elif config_dict['app_algorithm'] == 'singleclass':
        tif_files = [x for x in os.listdir(out_dir) if x.endswith('tif') and
                     not x[:3] in ['cla', 'pre', 'seg']]
    else:
        tif_files = [x for x in os.listdir(out_dir) if x.endswith('tif') and
                     not x[:3] in ['cla', 'pre', 'seg', 'pro']]

    clumps_gdf = vectorize_and_classify(clumps_tif,
                                        tif_files,
                                        out_dir,
                                        config_dict)
    SEGMENTATION_LOGGER.info("Classifying Segments")
    if config_dict['app_algorithm'] in US_ALGORITHMS:
        print(clumps_gdf.columns)
        clumps_gdf.columns = ['geometry', 'max']
        classification_polygons = clumps_gdf
    else:
        classification_polygons = classify_polygons(
            clumps_gdf,
            tif_files,
            config_dict['su_single_class_treshold'])

    # make multipolygons
    classification_polygons.geometry = \
        gdf_polygon_to_multipolygon(classification_polygons)

    # save
    classification_polygons.to_file(
        os.path.join(out_dir, 'segments_classified.gpkg'),
        driver='GPKG'
    )

    classification_polygons = gpd.read_file(
        os.path.join(out_dir, 'segments_classified.gpkg'))

    SEGMENTATION_LOGGER.info("Translating vectors back to tif")
    rasterize_polygons(classification_polygons,
                       clumps_tif,
                       out_dir,
                       config_dict)
