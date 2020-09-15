Welcome to SEPIA!
=================
.. automodule:: sepia
    :members:

.. image:: sepia.png
    :align: center
    :height: 200px
    :width: 160px
    :alt: Cuttlefish logo by Natalie Klein.



What is SEPIA?
==============

SEPIA (Simulation-Enabled Prediction, Inference, and Analysis) implements Bayesian emulation and calibration
with the ability to handle multivariate outputs. It is based on the Matlab code `GPMSA`_. For more, see :ref:`aboutsepia`.

Example code is described in :ref:`examples`. Full source code is available on `GitHub`_. While SEPIA is still under development,
users should pull the newest code from Github frequently. If you have installed using the
instructions below, you should not need to reinstall after pulling new code.

SEPIA is still under development and is intended to be a research tool, not production-level code.
Please submit questions as `GitHub`_ issues if you encounter unexpected behavior or need help using SEPIA.
As of 9/15/2020, basic functionality (model setup, likelihood evaluation, MCMC sampling,
multivariate calibration, visualization and diagnostics) is complete and has been tested.
Some features are not yet fully tested for all model cases (predictions, sensitivity analysis).

.. _GPMSA: https://github.com/lanl/gpmsa
.. _GitHub: https://github.com/lanl/SEPIA

Installation
============

First, pull down the source code from `GitHub`_ either by downloading a zip file or using `git clone`.
We recommend installing inside an Anaconda environment. The packages installed in the development environment
are listed in `environment.yml`. To create the environment, use::

        conda env create -f environment.yml

Activate the environment before installing sepia::

        source activate sepia

Then use the following command to install sepia::

        pip install -e .[sepia]


Resources for new users
=======================

Before getting started, we highly recommend reading the :ref:`workflow`. This goes through the general workflow
for setting up a model, doing MCMC, and checking results.

If you are familiar with the general workflow but need a quick reference for common tasks, see :ref:`helpful-code-snippets`.

If you are a `GPMSA`_ user, we suggest reading :ref:`sepia-for-gpmsa-users`.

Recent changes
==============

    * 9/10/20: users can pass custom theta constraint function `theta_fcon` to `SepiaModel` that evaluates True if theta follows the constraint
      and False otherwise. Must also pass in `theta_init` that satisfies the constraint.
    * 8/27/2020: predictions now expect `x` and `t` to be passed in the native (untransformed) space.
    * 8/25/2020: use `SepiaModel(data)` to set up model (no more `setup_model` function).

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
    optim
    predict
    plot
    sens

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
