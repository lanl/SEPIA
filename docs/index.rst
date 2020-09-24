Welcome to SEPIA!
=================
.. automodule:: sepia
    :members:

.. image:: sepia.png
    :align: center
    :height: 200px
    :width: 160px
    :alt: Cuttlefish logo by Natalie Klein.

.. image:: https://zenodo.org/badge/267692609.svg
   :target: https://zenodo.org/badge/latestdoi/267692609

What is SEPIA?
==============

SEPIA (Simulation-Enabled Prediction, Inference, and Analysis) implements Bayesian emulation and calibration
with the ability to handle multivariate outputs. It is based on the Matlab code `GPMSA`_. For more, see :ref:`aboutsepia`.

Example code is described in :ref:`examples`. Full source code is available on `GitHub`_. While SEPIA is still under development,
users should pull the newest code from Github frequently. If you have installed using the
instructions below, you should not need to reinstall after pulling new code.

SEPIA is still under development and is intended to be a research tool, not production-level code.
Please submit questions as `GitHub`_ issues if you encounter unexpected behavior or need help using SEPIA.
As of the current release, basic functionality (model setup, likelihood evaluation, MCMC sampling,
multivariate calibration, visualization and diagnostics, predictions) is complete and has been tested.

.. _GPMSA: https://github.com/lanl/gpmsa
.. _GitHub: https://github.com/lanl/SEPIA

Installation
============
For cleaner package management and to avoid conflicts between different versions of packages,
we recommend installing inside an Anaconda or pip environment (see `Conda docs`_ for details on Anaconda environments).
However, this is not required.

.. _Conda docs: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html

First, pull down the current source code from `GitHub`_ either by downloading a zip file or using `git clone`.
If you prefer, you can download the latest stable release instead of the master branch.

From the command line, while in the main SEPIA directory, use the following command to install sepia::

        pip install -e .[sepia]

The `-e` flag signals developer mode, meaning that if you update the code from Github, your installation will automatically
take those changes into account without requiring re-installation.
Some other essential packages used in SEPIA may be installed if they do not exist in your system or environment.

If you encounter problems with the above install method, you may try to install dependencies manually before installing SEPIA.
First, ensure you have a recent version of Python (greater than 3.5).
Then, install packages `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, and `tqdm`.

Resources for new users
=======================

Before getting started, we highly recommend reading the :ref:`workflow`. This goes through the general workflow
for setting up a model, doing MCMC, and checking results.

If you are familiar with the general workflow but need a quick reference for common tasks, see :ref:`helpful-code-snippets`.

If you are a `GPMSA`_ user, we suggest reading :ref:`sepia-for-gpmsa-users`.


Citing Sepia
============

    Using Sepia in your work? Please cite it as:

    James Gattiker, Natalie Klein, Earl Lawrence, & Grant Hutchings. lanl/SEPIA. Zenodo. https://doi.org/10.5281/zenodo.3979584

Pages
=====
.. toctree::
    :maxdepth: 2

    about
    examples
    quickstart
    snippets
    forgpmsa
    model_math

API
===
.. toctree::
    :maxdepth: 2

    data
    model
    shared_hier_model
    param
    prior
    mcmc
    model_internals
    predict
    plot
    sens

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
