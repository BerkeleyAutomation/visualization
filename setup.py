"""
Visualization setup file.
"""
from setuptools import setup

requirements = ["imageio", "numpy", "matplotlib", "trimesh[easy]", "autolab_core", "pyrender"]

exec(open("visualization/version.py").read())


setup(
    name="visualization",
    version=__version__,
    description="Visualization toolkit for the Berkeley AutoLab.",
    long_description="Visualization toolkit for the Berkeley AutoLab.",
    author="Matthew Matl",
    author_email="mmatl@eecs.berkeley.edu",
    license="Apache Software License",
    url="https://github.com/BerkeleyAutomation/visualization",
    keywords="robotics visualization rendering 3D",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=["visualization"],
    install_requires=requirements,
    extras_require={"docs": ["sphinx", "sphinxcontrib-napoleon", "sphinx_rtd_theme"]},
)
