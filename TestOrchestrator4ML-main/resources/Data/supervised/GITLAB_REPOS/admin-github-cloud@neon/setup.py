#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import os
from setuptools import setup, find_packages, Command
import subprocess

# Define version information
VERSION = '1.5.4'
FULLVERSION = VERSION
write_version = True

try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip().decode("utf-8")
except:
    pass

if write_version:
    txt = "# " + ("-" * 77) + "\n"
    txt += "# Copyright 2016 Nervana Systems Inc.\n"
    txt += "# Licensed under the Apache License, Version 2.0 "
    txt += "(the \"License\");\n"
    txt += "# you may not use this file except in compliance with the "
    txt += "License.\n"
    txt += "# You may obtain a copy of the License at\n"
    txt += "#\n"
    txt += "#      http://www.apache.org/licenses/LICENSE-2.0\n"
    txt += "#\n"
    txt += "# Unless required by applicable law or agreed to in writing, "
    txt += "software\n"
    txt += "# distributed under the License is distributed on an \"AS IS\" "
    txt += "BASIS,\n"
    txt += "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or "
    txt += "implied.\n"
    txt += "# See the License for the specific language governing permissions "
    txt += "and\n"
    txt += "# limitations under the License.\n"
    txt += "# " + ("-" * 77) + "\n"
    txt += "\"\"\"\n%s\n\"\"\"\nVERSION = '%s'\nSHORT_VERSION = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), 'neon', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()


setup(name='neon',
      version=VERSION,
      description="Nervana's deep learning framework",
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='info@nervanasys.com',
      url='http://www.nervanasys.com',
      license='License :: OSI Approved :: Apache Software License',
      scripts=['bin/neon', 'bin/nvis'],
      packages=find_packages(),
      package_data={'neon': ['backends/kernels/sass/*.sass',
                             'backends/kernels/cubin/*.cubin',
                             'backends/kernels/maxas/*.pl',
                             'backends/kernels/maxas/MaxAs/*.pm',
                             '../loader/bin/*.so']},
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Environment :: Console :: Curses',
                   'Environment :: Web Environment',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: ' +
                   'Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: System :: Distributed Computing'])
