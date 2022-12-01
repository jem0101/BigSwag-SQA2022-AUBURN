"""Vector utilities"""
import logging

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon

UTILS_VECTOR_LOGGER = logging.getLogger(__name__)

def vectorize_array(array_to_vectorize, transform=None):
    """Vectorize a tif file and return geodataframe.

    Args
        array_to_vectorize (path):a rray to vectorize_and_classify

    Returns:
        geodataframe with polygons

    """
    polygonized = shapes(array_to_vectorize,
                         connectivity=8,
                         transform=transform)
    shapelist, valuelist = zip(*polygonized)
    shapelist = [Polygon(x['coordinates'][0]) for x in shapelist]
    new_poly_dict = {'geometry': shapelist, 'segment_id': valuelist}

    return gpd.GeoDataFrame.from_dict(new_poly_dict)


def get_subset_of_polygons(window, transform, polygons, col_name='max'):
    """Get a subset of a geodataframe from a rasterio window

    Uses a rasterio window to make an intersection on the whole polygon
    dataframe.

    Args:
        window (rasterio.Window): window of interest
        transform (dict):   transform of the window
        polygons (GeoDataFrame): polygons to classify
        col_name (str): Name of the column to subset

    Returns:
        list of [geometry, value] pairs that can be consumed by GeoPandas


    """
    left, bottom, right, top = rasterio.windows.bounds(window,
                                                       transform)

    window_polygon = Polygon([[left, bottom],
                              [left, top],
                              [right, top],
                              [right, bottom]])

    window_df = gpd.GeoDataFrame({'geometry': [window_polygon],
                                  'name': [1]})
    window_df.crs = {'init': 'epsg:4326'}

    subset_polygons = gpd.overlay(window_df,
                                  polygons,
                                  how='intersection')

    return list(zip(list(subset_polygons['geometry'].values),
                    list(subset_polygons[col_name].values)))


def gdf_polygon_to_multipolygon(gdf):
    """Converts all geometries to multipolyon, so there are no issues when
    saving these to a file and adds a crs (epsg:4326) to the gdf

        Args:
            gdf (GeoDataFrame): Containing polygons

        Returns:
            gdf (GeoDataFrame): GDF with all polygons converted to multipolygons

    """
    geometry = [MultiPolygon([feature]) if isinstance(feature, Polygon)
                else feature for feature in gdf["geometry"]]
    return geometry
