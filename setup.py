"""
Setup of core python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'numpy',
    'matplotlib',
    #'meshrender'
]
setup(name='visualization',
      version='0.1.0',
      description='AutoLab visualization code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': '.'},
      packages=['visualization'],
      install_requires=requirements,
     )

