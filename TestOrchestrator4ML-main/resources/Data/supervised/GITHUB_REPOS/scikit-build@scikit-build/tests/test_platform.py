#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_platform
----------------------------------

Tests for platforms, to verify that CMake correctly does a test compilation.
"""

import os
import sys
import platform
import pytest

from skbuild.platform_specifics import get_platform
from skbuild.utils import mkdir_p

# XXX This should probably be a constant imported from skbuild.constants
test_folder = "_cmake_test_compile"

# skbuild_platform is shared across each test.  It's a platform-specific object
# that defines default CMake generator strings.
skbuild_platform = get_platform()


def test_platform_has_entries():
    assert(len(skbuild_platform.default_generators) > 0)


def test_write_compiler_test_file():
    # write the file that CMake will use to test compile (empty list indicates
    # we're testing no languages.)
    skbuild_platform.write_test_cmakelist([])
    try:
        # verify that the test file exists (it's not valid, because it has no
        # languages)
        assert(os.path.exists(os.path.join(test_folder, "CMakeLists.txt")))
    finally:
        skbuild_platform.cleanup_test()


def test_cxx_compiler():

    # Create a unique subdirectory 'foo' that is expected to be removed.
    test_build_folder = os.path.join(test_folder, 'build', 'foo')
    mkdir_p(test_build_folder)

    generator = skbuild_platform.get_best_generator(languages=["CXX", "C"],
                                                    cleanup=False)
    # TODO: this isn't a true unit test.  It depends on the test CMakeLists.txt
    #       file having been written correctly.
    # with the known test file present, this tries to generate a makefile
    # (or solution, or whatever).
    # This test verifies that a working compiler is present on the system, but
    # doesn't actually compile anything.
    try:
        assert(generator is not None)
        assert not os.path.exists(test_build_folder)
    finally:
        skbuild_platform.cleanup_test()


@pytest.mark.skipif(platform.system().lower() in ["darwin", "windows"],
                    reason="no fortran compiler is available by default")
@pytest.mark.fortran
def test_fortran_compiler():
    generator = skbuild_platform.get_best_generator(languages=["Fortran"])
    # TODO: this isn't a true unit test.  It depends on the test
    #       CMakeLists.txt file having been written correctly.
    # with the known test file present, this tries to generate a
    # makefile (or solution, or whatever).
    # This test verifies that a working compiler is present on the system, but
    # doesn't actually compile anything.
    try:
        assert(generator is not None)
    finally:
        skbuild_platform.cleanup_test()


def test_generator_cleanup():
    # TODO: this isn't a true unit test.  It is checking that none of the
    # other tests have left a mess.
    assert(not os.path.exists(test_folder))


@pytest.mark.parametrize("supported_platform",
                         ['darwin', 'freebsd', 'linux', 'windows', 'os400'])
def test_known_platform(supported_platform, mocker):
    mocker.patch('platform.system', return_value=supported_platform)
    platforms = {
        'freebsd': 'BSD',
        'linux': 'Linux',
        'darwin': 'OSX',
        'windows': 'Windows',
        'os400': 'BSD'
    }
    expected_platform_classname = "%sPlatform" % platforms[supported_platform]
    assert get_platform().__class__.__name__ == expected_platform_classname


def test_unsupported_platform(mocker):
    mocker.patch('platform.system', return_value='bogus')

    failed = False
    message = ""
    try:
        get_platform()
    except RuntimeError as e:
        failed = True
        message = str(e)

    assert failed
    assert "Unsupported platform: bogus." in message


@pytest.mark.skipif(sys.platform != 'win32', reason='Requires Windows')
def test_cached_generator():
    platform = get_platform()
    generator = platform.get_generator('Ninja')
    env = generator.env

    assert 'Visual Studio' in env['LIB'] or 'Visual C++' in env['LIB']
