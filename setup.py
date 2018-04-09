"""
Visualization setup file.
"""
from setuptools import setup

requirements = [
    'imageio',
    'numpy',
    'matplotlib',
    'trimesh[easy]',
    'autolab_core',
    'autolab_perception',
    'meshrender'
]

exec(open('visualization/version.py').read())


setup(
    name='visualization',
    version = __version__,
    description = 'Visualization toolkit for the Berkeley AutoLab.',
    long_description = 'Visualization toolkit for the Berkeley AutoLab.',
    author = 'Matthew Matl',
    author_email = 'mmatl@eevs.berkeley.edu',
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
        ],
        'full' : [
            'meshpy',
        ],
    }
)
