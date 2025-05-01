#!/usr/bin/env python3
from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['hexapod_rl'],
    package_dir={'': 'src'},
)

setup(**setup_args)
