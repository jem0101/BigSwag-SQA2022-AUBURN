#!/usr/bin/env python
import sys
import os

from distutils.dist import Distribution

if 'setuptools' in sys.modules or any(
        s.startswith('bdist') for s in sys.argv) or any(
        s.startswith('develop') for s in sys.argv):
    from setuptools import setup as setup
    from setuptools import Extension
else:  # special case for runtests.py
    from distutils.core import setup as setup
    from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
# Don't use cython if this is a distributed release without .pyx files
if not os.path.isfile('sima/motion/_motion.pyx'):
    USE_CYTHON = False

# Avoid installing setup_requires dependencies if the user just
# queries for information
if (any('--' + opt in sys.argv for opt in
        Distribution.display_option_names + ['help']) or
        'clean' in sys.argv):
    setup_requires = []
else:
    setup_requires = ['numpy']


# --- Encapsulate NumPy imports in a specialized Extension type ---------------

# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(Extension, object):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        super(NumpyExtension, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs


extensions = [
    NumpyExtension(
        'sima.motion._motion',
        sources=['sima/motion/_motion.%s' % ('pyx' if USE_CYTHON else 'c')],
        include_dirs=[],
    ),
    NumpyExtension(
        'sima.segment._opca',
        sources=['sima/segment/_opca.%s' % ('pyx' if USE_CYTHON else 'c')],
        include_dirs=[],
    )
]

if USE_CYTHON:
    extensions = cythonize(extensions)


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="sima",
    version="1.3.2",
    packages=['sima',
              'sima.misc',
              'sima.motion',
              'sima.motion.tests',
              'sima.segment',
              'sima.segment.tests',
              ],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.13.0',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
        'scikit-learn>=0.11',
        'pillow>=2.6.1',
        'future>=0.14',
    ],
    package_data={
        'sima': [
            'tests/*.py',
            'tests/data/example.sima/*',
            'tests/data/example.tif',
            'tests/data/example.h5',
            'tests/data/example-volume.h5',
            'tests/data/imageJ_ROIs.zip',
            'tests/data/example-tiffs/*.tif',
        ]
    },
    #
    # metadata for upload to PyPI
    author="The SIMA Development Team",
    author_email="software@losonczylab.org",
    description="Software for analysis of sequential imaging data",
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience segmentation",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    ext_modules=extensions,
    setup_requires=setup_requires,
    # setup_requires=['setuptools_cython'],
    url="http://www.losonczylab.org/sima/",
    platforms=["Linux", "Mac OS-X", "Windows"],
    #
    # could also include long_description, download_url, classifiers, etc.
)
