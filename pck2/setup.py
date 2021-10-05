#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pyloric, a simulator published under MIT license.

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "pyloric"
DESCRIPTION = "Simulator of the pyloric network in the STG."
KEYWORDS = "pyloric stg cython simulator"
URL = "https://github.com/mackelab/pyloric"
EMAIL = "michael.deistler@uni-tuebingen.de"
AUTHOR = "Michael Deistler"
REQUIRES_PYTHON = ">=3.6.0"

REQUIRED = [
    "tqdm",
    "cython",
    "numpy",
    "pandas",
    "sbi",
    "scipy",
    "svgutils",
    "torchdiffeq",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name=NAME,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license="AGPLv3",
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
