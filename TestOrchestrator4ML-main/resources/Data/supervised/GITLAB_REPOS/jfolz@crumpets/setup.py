import os.path as pt

from setuptools import setup
from setuptools import find_packages

from version import find_version
from version import find_file_version
from version import write_version


PACKAGE_DIR = pt.abspath(pt.dirname(__file__))


packages = find_packages(
    include=['crumpets', 'crumpets.*'],
    exclude=['crumpets.torch.*', 'crumpets.torch']
)


package_data = {
    package: [
        '*.py',
        '*.txt',
        '*.json'
    ]
    for package in packages
}


def read_requirements():
    with open(pt.join(PACKAGE_DIR, 'requirements.txt')) as f:
        return [l.strip(' \n') for l in f]


dependencies = read_requirements()


scripts = []


old_version = find_file_version('crumpets', '__init__.py')
version = find_version('crumpets', '__init__.py')
# update file to match wheel version
write_version(version, 'crumpets', '__init__.py')


setup(
    name='crumpets',
    version=version,
    author='Joachim Folz',
    author_email='joachim.folz@dfki.de',
    python_requires='>3.5.2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='crumpets deep-learning saliency prediction',
    packages=packages,
    package_data=package_data,
    setup_requires=['numpy>=1.15.0'],
    install_requires=dependencies,
    scripts=scripts,
    ext_modules=[],
    zip_safe=False,
)


# restore previous state of the local file
write_version(old_version, 'crumpets', '__init__.py')
