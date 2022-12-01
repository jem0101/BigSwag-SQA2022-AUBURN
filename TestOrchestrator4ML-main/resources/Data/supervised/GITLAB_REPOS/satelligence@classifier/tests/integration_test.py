"Testing module of classifier"
import json
import os

from click.testing import CliRunner
import numpy as np
import pytest
import rasterio

from classifier.cli import classification
from classifier.settings import WORKSPACE

@pytest.fixture(scope="module")
def runner():
    """Click Cli Runner for tests"""
    return CliRunner()

def make_config(**kwargs):
    """Make a config file for a specific test
    Args:
        The parameters to change as named variables

    """
    with open(os.path.join(WORKSPACE, 'config.json'), 'w') as dst:
        dst.write(json.dumps(kwargs))

def setup():
    """
    Setup for the tests
    Returns:

    """

    nowhere = 0.0
    pixel_size = 2.0

    transform = rasterio.Affine(
        pixel_size,
        nowhere,
        nowhere,
        nowhere,
        -pixel_size,
        nowhere
    )

    raster = os.path.join(WORKSPACE, 'test_input.tif')
    data = np.array([
        [
            [1, 200],
            [200, 1]
        ],
        [
            [4, 110],
            [100, 3]
        ]
    ], dtype=np.uint8)

    with rasterio.open(
        raster,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype='uint8',
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(data)

    return raster


# pylint:disable=redefined-outer-name
def test_unsupervised(runner):
    """
    Test for running a simple unsupervised classification
    Returns:

    """
    raster = setup()
    make_config(app_threads=1)
    runner.invoke(classification, ["--name", 'test', raster])
    directory = os.path.join(WORKSPACE, 'test')
    with rasterio.open(
        os.path.join(directory, 'classification.tif'),
        'r',
    ) as dst:
        result = dst.read(1)
        assert result[0][0] == 0
        assert result[0][1] == 1
