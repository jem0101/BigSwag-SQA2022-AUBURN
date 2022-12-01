"""Module setup."""

import os
import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "aries_cloudagent"
version_meta = runpy.run_path("./{}/version.py".format(PACKAGE_NAME))
VERSION = version_meta["__version__"]


with open(os.path.abspath("./README.md"), "r") as fh:
    long_description = fh.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        include_package_data=True,
        package_data={"aries_cloudagent": ["requirements.txt"]},
        install_requires=parse_requirements("requirements.txt"),
        tests_require=parse_requirements("requirements.dev.txt"),
        extras_require={"indy": parse_requirements("requirements.indy.txt")},
        python_requires=">=3.6.3",
        scripts=["bin/aca-py"],
    )
