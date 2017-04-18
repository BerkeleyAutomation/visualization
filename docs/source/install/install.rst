Installation Instructions
=========================

Dependencies
~~~~~~~~~~~~
The `visualization` module uses `matplotlib`_ for 2D visualization
and `mayavi`_ for 3D visualization, which you can install with pip.
The `visualization` module also depends on the Berkeley AutoLab's `core`_ and
`perception`_ modules, which can be installed from github.

.. _matplotlib: http://www.matplotlib.org/
.. _mayavi: http://docs.enthought.com/mayavi/mayavi/
.. _core: https://github.com/mmatl/core
.. _perception: https://github.com/mmatl/perception

Any other dependencies will be installed automatically when `visualization` is
installed with `pip`.

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~
You can clone or download our source code from `Github`_. ::

    $ git clone git@github.com:jeffmahler/visualization.git

.. _Github: https://github.com/jeffmahler/visualization

Installation
~~~~~~~~~~~~
To install `visualization` in your current Python environment, simply
change directories into the `visualization` repository and run ::

    $ pip install -e .

or ::

    $ pip install -r requirements.txt

Alternatively, you can run ::

    $ pip install /path/to/visualization

to install `visualization` from anywhere.

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~
Building `visualization`'s documentation requires a few extra dependencies --
specifically, `sphinx`_ and a few plugins.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying Documentation
~~~~~~~~~~~~~~~~~~~~~~~
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.
