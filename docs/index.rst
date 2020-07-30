Welcome to Sepia!
=================
.. automodule:: sepia
    :members:

.. image:: sepia.png
    :align: center
    :height: 200px
    :width: 160px

.. _GPMSA: https://github.com/lanl/gpmsa

.. topic:: What is Sepia?

    Sepia is a Python implementation of `GPMSA`_. For more, see :ref:`aboutsepia`.

    Sepia implements Bayesian emulation and calibration with the ability to handle multivariate outputs.

    Example jupyter notebooks are found `here`_. These self-contained examples may be helpful for setting up your code.

    Full source code on `GitHub`_.

    Warning: Sepia is still in development. Basic functionality (model setup, likelihood evaluation, MCMC sampling,
    multivariate calibration, predictions) is complete and has been tested. Some features are untested or still being
    developed (visualization and diagnostics, sensitivity analysis).

    Users should pull the newest code frequently, particularly if you encounter errors. If you have installed using the
    instructions on GitHub, you should not need to reinstall after pulling new code.

.. _here: https://github.com/lanl/SEPIA/tree/master/sepia/Examples

.. _GitHub: https://github.com/lanl/SEPIA

.. topic:: Read before starting: general workflow

    Before getting started, we highly recommend reading the :ref:`workflow`.

.. topic:: Code snippets

    Familiar with the general workflow but looking for some tips on how to do common tasks? See :ref:`helpful-code-snippets`.

.. topic:: Installation

    First, pull down the source code from `GitHub`_.
    We recommend installing inside an Anaconda environment. The packages installed in the development environment
    are listed in `environment.yml`. Use `conda env create -f environment.yml` to create the environment, then activate as
    `source activate sepia` before installing sepia.

    Then use the following command to install sepia:

    `pip install -e .[sepia]`


Pages
*****
.. toctree::
    :maxdepth: 2

    about
    quickstart
    snippets

API
***
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



Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
