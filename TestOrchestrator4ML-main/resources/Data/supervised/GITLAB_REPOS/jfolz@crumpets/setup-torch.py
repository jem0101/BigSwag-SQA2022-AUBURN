import os.path as pt

from setuptools import setup
from setuptools import find_packages

from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension

from version import find_file_version
from version import find_cuda_version
from version import find_version
from version import write_version


PACKAGE_DIR = pt.abspath(pt.dirname(__file__))


def make_augmentation_extension():
    return CUDAExtension(
        name='crumpets.torch._augmentation_cuda',
        sources=['crumpets/torch/_augmentation.cu']
    )


# define extensions
ext_modules = [
    make_augmentation_extension()
]


packages = find_packages(
    include=['crumpets.torch', 'crumpets.torch.*'],
)


package_data = {
    package: [
        '*.py',
        '*.txt',
        '*.json'
    ]
    for package in packages
}


scripts = []


old_version = find_file_version('crumpets', 'torch', '__init__.py')
cuda_version = find_cuda_version()
crumpets_version = find_version('crumpets', '__init__.py')
version = find_version('crumpets', '__init__.py', cuda=cuda_version)
# update file to match wheel version
write_version(version, 'crumpets', 'torch', '__init__.py')


# depend on current crumpets version
with open(pt.join(PACKAGE_DIR, 'requirements-torch.txt')) as f:
    dependencies = [l.strip(' \n') for l in f]
dependencies = [d for d in dependencies if not d.startswith('crumpets')]
dependencies.append('crumpets==%s' % crumpets_version)


setup(
    name='crumpets-torch',
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
        'Programming Language :: Python :: 3.8',
    ],
    keywords='crumpets torch pytorch augmentation',
    packages=packages,
    package_data=package_data,
    setup_requires=['numpy>=1.15.0'],
    install_requires=dependencies,
    scripts=scripts,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)


write_version(old_version, 'crumpets', 'torch', '__init__.py')
