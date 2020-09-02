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

    .. note:: Warning: Sepia is still in development. Basic functionality (model setup, likelihood evaluation, MCMC sampling,
        multivariate calibration, visualization and diagnostics) is complete and has been tested.
        Some features are not fully tested for all model cases or still being developed (predictions, sensitivity analysis).

        Users should pull the newest code frequently, particularly if you encounter errors. If you have installed using the
        instructions on GitHub, you should not need to reinstall after pulling new code.

.. _here: https://github.com/lanl/SEPIA/tree/master/examples

.. _GitHub: https://github.com/lanl/SEPIA

.. topic:: Installation

    First, pull down the source code from `GitHub`_.
    We recommend installing inside an Anaconda environment. The packages installed in the development environment
    are listed in `environment.yml`. To create the environment, use::

        conda env create -f environment.yml

    Activate the environment before installing sepia::

        source activate sepia

    Then use the following command to install sepia::

        pip install -e .[sepia]

.. topic:: Read first: general workflow

    Before getting started, we highly recommend reading the :ref:`workflow`. This goes through the general workflow
    for setting up a model, doing MCMC, and checking results.

.. topic:: Code snippets

    Familiar with the general workflow but looking for some tips on how to do common tasks? See :ref:`helpful-code-snippets`.

.. topic:: Recent changes

    * 8/27/2020: predictions now expect `x` and `t` to be passed in the native (untransformed) space.
    * 8/25/2020: use `SepiaModel(data)` to set up model (no more `setup_model` function).

.. topic:: Citing Sepia

    Using Sepia in your work? Cite as:

    James Gattiker, Natalie Klein, Earl Lawrence, & Grant Hutchings. lanl/SEPIA. Zenodo. https://doi.org/10.5281/zenodo.3979584

.. topic:: How should one pronounce Sepia?

    This is a topic of some debate. `This`_ video suggests "see-pee-uh".

.. _This: https://www.youtube.com/watch?v=qUi1F254t6M

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
    plot



Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
