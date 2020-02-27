"""
Visualization setup file.
"""
from setuptools import setup
import sys

def get_imageio_dep():
    if sys.version[0] == "2":
        return 'imageio<=2.6.1'
    return 'imageio'

requirements = [
    get_imageio_dep(),
    'numpy',
    'matplotlib<=2.2.0',
    'trimesh[easy]',
    'autolab_core',
    'autolab_perception',
    'pyrender'
]

exec(open('visualization/version.py').read())


setup(
    name='visualization',
    version = __version__,
    description = 'Visualization toolkit for the Berkeley AutoLab.',
    long_description = 'Visualization toolkit for the Berkeley AutoLab.',
    author = 'Matthew Matl',
    author_email = 'mmatl@eecs.berkeley.edu',
    license = 'Apache Software License',
    url = 'https://github.com/BerkeleyAutomation/visualization',
    keywords = 'robotics visualization rendering 3D',
    classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages = ['visualization'],
    install_requires = requirements,
    extras_require = { 'docs' : [
            'sphinx',
            'sphinxcontrib-napoleon',
            'sphinx_rtd_theme'
        ]
    }
)
