# -*- coding: utf-8 -*-
# (c) Satelligence, see LICENSE.
# pylint: skip-file
from setuptools import setup
import os

version = '2.4.1.dev0'

long_description = open('README.md').read()

test_requirements = [
    'pytest'
]

setup(
    name='s11-classifier',
    version=version,
    description="Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rens Masselink",
    author_email='rens.masselink@satelligence.com',
    url='https://gitlab.com/satelligence/classifier',
    packages=[
        'classifier',
    ],
    package_dir={'classifier':
                 'classifier'},
    include_package_data=True,
    install_requires=[
        'Fiona>=1.8.3, <1.9.0',
        'matplotlib>=2.0.0, <2.1.0',
        'numpy>=1.13.1, <1.14.0',
        'pandas>=0.22.0, <0.23.0',
        'pylint>=1.8.2, <1.9.0',
        'pytest>=5.2.0, <5.3.0',
        'rasterio>=1.0.11, <1.1.0',
        'scikit_learn>=0.21.3, <0.22.0',
        'xgboost>=0.81, <0.82',
        'boto3>=1.9.4, <1.10.0',
        'folium>=0.6.0, <0.7.0',
        'geopandas>=0.4.0, <0.5.0',
        'geojson>=2.4.0, <2.5.0',
        'click>=6.7, <6.8.0',
        'rasterstats>=0.13.0, <0.14.0',
        'rtree>=0.8.3, <0.9.0'
    ],
    license="Apache-2.0",
    zip_safe=False,
    python_requires='>=3.5'
)
