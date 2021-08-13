from setuptools import setup, find_packages
import os

__version__ = '0.0.1'
NAME = 'DeepMice'
AUTHORS = ''
MAINTEINERS = ''

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    author=AUTHORS,
    maintainer=MAINTEINERS,
    description='',
    license='BSD 3',
    install_requires=['numpy',
                      'scipy',
                      'xarray',
                      'netcdf4']
)
