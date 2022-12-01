# -*- coding: utf-8 -*-
#
# first steps documentation build configuration file, created by
# sphinx-quickstart on Fri Oct  8 15:09:05 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from conf import *

TITLE = "%s v%s" % ("First Steps in Orange Canvas", VERSION)

html_title = TITLE
epub_title = TITLE

latex_documents = [
    ('index', 'reference.tex', TITLE,
     AUTHOR, 'manual'),
    ]

