import os.path as pt
import re
from subprocess import check_output
from subprocess import CalledProcessError


PACKAGE_DIR = pt.abspath(pt.dirname(__file__))
VERSION_PATTERN = r"^__version__ = ['\"]([^'\"]*)['\"]"
CUDA_PATTERN = r"release ([0-9.]+),"


def find_file_version(*file_paths):
    # read the version file
    version_path = pt.join(PACKAGE_DIR, *file_paths)
    with open(version_path, encoding='utf8') as f:
        version_file = f.read()
    # match version string
    version_match = re.search(VERSION_PATTERN, version_file, re.MULTILINE)
    if version_match:
        return version_match.group(1)


def find_git_version(fmt='{tag}+{cuda}dev{commitcount}.git{sha}{dirty}', cuda=None):
    try:
        cmd = 'git', 'describe', '--tags', '--long', '--dirty'
        description = check_output(cmd).decode('utf-8').strip()
    except (FileNotFoundError, CalledProcessError):
        return None
    parts = description.split('-')
    assert len(parts) in (3, 4)
    dirty = parts[-1] == 'dirty'
    tag, count, sha = parts[:3]
    if count == '0' and not dirty:
        return tag + ('+cuda' + cuda if cuda else '')
    return fmt.format(
        tag=tag,
        cuda='cuda' + cuda + '.' if cuda else '',
        commitcount=count,
        sha=sha.lstrip('g'),
        dirty='.dirty' if dirty else ''
    )


def find_version(*file_paths, cuda=None):
    """
    Find package version based on
    `setuptools git versioning
     <https://github.com/pyfidelity/setuptools-git-version>`_
    method and fall back to
    `pip's single-source version
     <https://python-packaging-user-guide.readthedocs.io/single_source_version/>`_
    method if not a git repository.

    :param file_paths:
        parts of the path to the file that defines the version;
        joined with os.path.join
    :param cuda:
        optional, add CUDA version to output
    :return:
        git or file version string
    :raise:
        RuntimeError if neither version could be found
    """
    git_version = find_git_version(cuda=cuda)
    if git_version is not None:
        return git_version

    file_version = find_file_version(*file_paths)
    if file_version is not None:
        return file_version + ('+cuda' + cuda if cuda else '')

    raise RuntimeError("Unable to find version.")


def write_version(version, *file_paths):
    """
    Replace "__version__ = '...'" with the given version string
    in a given file on the filesystem.

    :param version:
        replacement version string
    :param file_paths:
        paths to file where version is replaced, joined by os.path
    """
    version_path = pt.join(PACKAGE_DIR, *file_paths)
    with open(version_path, encoding='utf8') as f:
        version_file = f.read()
    new_version_file = re.sub(
        VERSION_PATTERN,
        "__version__ = '%s'" % version,
        version_file,
        flags=re.MULTILINE
    )
    with open(version_path, 'w', encoding='utf-8') as f:
        f.write(new_version_file)


def find_cuda_version():
    """
    Use `nvcc --version` output to find CUDA version.
    Dots are removed from version string.

    :return: normalized CUDA version string, e.g. `'101'` for version 10.1
    :raise: `RuntimeError` if nvcc is not on path
    """
    try:
        # example:
        # $ nvcc --version
        # nvcc: NVIDIA (R) Cuda compiler driver
        # Copyright (c) 2005-2018 NVIDIA Corporation
        # Built on Sat_Aug_25_21:08:01_CDT_2018
        # Cuda compilation tools, release 10.0, V10.0.130
        cmd = 'nvcc', '--version'
        nvcc_output = check_output(cmd).decode('utf-8').strip()
        match = re.search(CUDA_PATTERN, nvcc_output)
        if match:
            return match.group(1).replace('.', '')
    except (FileNotFoundError, CalledProcessError):
        raise RuntimeError('cannot run nvcc, CUDA not available/not on path?')
