#!/usr/bin/env python3
from setuptools import setup

setup(name     = 'avalanche_ml',
      #author   = '',
      #description = ''
      version  = '0.7',
      license  = 'MIT',
      url      = 'https://github.com/NVE/avalanche_ml',

      packages = ['analysis',
                  'avaml',
                  'avaml.machine',
                  'modeling',
                  'utils'],

      install_requires = ['numpy>1.17.0', 'pandas>1.0.0', 'matplotlib>3.0.0']
     )
