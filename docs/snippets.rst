.. _helpful-code-snippets:

Helpful Code Snippets
=====================

Unlike notebooks, these are not self-contained examples, but are meant to be a quick reference for specific tasks.

:ref:`SepiaData inputs`

:ref:`SepiaData operations`

:ref:`Setup model and customize`


SepiaData inputs
----------------

:ref:`sepiadata` objects are used to hold various types of data inputs to a model.
The types and sizes of each input to `SepiaData` helps Sepia determine which kind of model is to be set up.

A Sepia model may contain only simulation data (an emulator-only model) or both simulation and observed data.

For an emulator-only model, the possible inputs are:

* `x_sim`: controllable simulation inputs (those inputs that would also be known for observed data).
  These are optional. If not provided, Sepia internally uses a dummy set of controllable input values all equal to 0.5.
* `t_sim`: simulation inputs that would not be known for observed data.
  At least one of `x_sim` or `t_sim` must be provided to make a valid model.
* `y_sim`: simulation outputs, must be provided.
* `y_ind_sim`: vector of indices for multivariate simulation output, required if the output is multivariate.

If, in addition, observed data will be included, the following possible inputs would be included:

* `x_obs`: controllable inputs for the observed data, which are again optional if there are none for your data.
* `y_obs`: observation outputs, must be provided.
* `y_ind_obs`: indices for multivariate observation outputs, required if the outputs are multivariate.

It is always good to call `print(data)` on your `SepiaData` object to verify your setup is as intended.

Examples of data setup for different kinds of Sepia models (see :ref:`sepiadata` for fuller explanation of inputs):

Univariate-output emulator-only data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    data = SepiaData(t_sim=t, y_sim=y)           # No controllable input
    data = SepiaData(x_sim=x, y_sim=y)           # Only controllable input
    data = SepiaData(x_sim=x, t_sim=t, y_sim=y)  # Controllable and other inputs

Multivariate-output emulator-only data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    data = SepiaData(t_sim=t, y_sim=y, y_ind_sim=y_ind)           # No controllable input
    data = SepiaData(x_sim=x, y_sim=y, y_ind_sim=y_ind)           # Only controllable input
    data = SepiaData(x_sim=x, t_sim=t, y_sim=y, y_ind_sim=y_ind)  # Controllable and other inputs

Univariate-output simulation and observed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    data = SepiaData(t_sim=t, y_sim=y, y_obs=y_obs)                        # No controllable input
    data = SepiaData(x_sim=x, t_sim=t, y_sim=y, x_obs=x_obs, y_obs=y_obs)  # Controllable and other inputs

Multivariate-output simulation and observed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    data = SepiaData(t_sim=t, y_sim=y, y_ind_sim=y_ind,
                     y_obs=y_obs, y_ind_obs=y_ind_obs)               # No controllable input
    data = SepiaData(x_sim=x, t_sim=t, y_sim=y, y_ind_sim=y_ind,
                     x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_ind_obs)  # Controllable and other inputs


SepiaData operations
--------------------

Regardless of the inputs given to `SepiaData`, there are a few key methods which generally should be called before
setting up the model.

First, we want to transform `x` and `t` inputs to the unit hypercube::

    data.transform_xt()

Next, we want to standardize the `y` outputs::

    data.standardize_y()

If the outputs are multivariate, we want to set up a principal component (PC) basis and optionally, a discrepancy basis::

    # PC basis
    data.create_K_basis(n_pc=5)     # With 5 PCs
    data.create_K_basis(n_pc=0.99)  # Enough PCs for at least 99 pct variance explained

    # Discrepancy basis -- optional
    data.create_D_basis(type='linear')  # Default linear discrepancy
    data.create_D_basis(D=D)            # Pass in custom D basis


Setup model and customize
-------------------------

Once the data structure is set up correctly, the inputs are in the unit hypercube, the outputs are standardized,
and basis vectors are created (for multivariate output), we are ready to set up the Sepia model::

    from SepiaModelSetup import setup_model
    model = setup_model(data)

The model parses the `SepiaData` structure to understand what kind of model is being set up and does a lot of
precomputation of various quantities to prepare for likelihood evaluations.
It also sets up default priors, MCMC step types and step sizes, and default starting values for MCMC.

These can be customized after calling `setup_model`.

Custom start values
^^^^^^^^^^^^^^^^^^^

Each parameter in the model has attribute `val` which holds the start values (or, during MCMC, the current values).
Prior to running MCMC, these can be directly modified using the `set_val` method::

    # Single scalar applies to all thetas
    model.params.theta.set_val(0.7)
    # Or pass an array of shape model.params.theta.val_shape
    model.params.theta.set_val(np.array([[0.7, 0.5, 0.1]]))