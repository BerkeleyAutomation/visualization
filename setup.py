"""
Setup of core python codebase
Author: Jeff Mahler
"""
from setuptools import setup

setup(name='visualization',
      version='0.1.dev0',
      description='AutoLab visualization code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': '.'},
      packages=['visualization'],
      #test_suite='test'
     )

