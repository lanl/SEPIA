.. _helpful-code-snippets:

Helpful Code Snippets
=====================

Unlike notebooks, these are not self-contained examples, but are meant to be a quick reference for specific tasks.
These are not necessarily exhaustive examples; see full class documentation for all possible arguments and options.

:ref:`SepiaData inputs`

:ref:`SepiaData operations`

:ref:`Set up model`

:ref:`Customize and run MCMC`

:ref:`Extracting MCMC samples`

:ref:`Making predictions`

:ref:`Hierarchical or shared theta models`

:ref:`SepiaPlot visualization utilities`


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


Set up model
------------

Once the data structure is set up correctly, the inputs are in the unit hypercube, the outputs are standardized,
and basis vectors are created (for multivariate output), we are ready to set up the Sepia model::

    from SepiaModelSetup import setup_model
    model = setup_model(data)
    print(model)

The model parses the `SepiaData` structure to understand what kind of model is being set up and does a lot of
precomputation of various quantities to prepare for likelihood evaluations.
It also sets up default priors, MCMC step types and step sizes, and default starting values for MCMC.


Customize and run MCMC
----------------------

After calling `setup_model`, various aspects of the MCMC can be customized prior to doing MCMC.

Custom start values
^^^^^^^^^^^^^^^^^^^

Each parameter in the model has attribute `val` which holds the start values (or, during MCMC, the current values).
Prior to running MCMC, these can be directly modified using the `set_val` method::

    # Single scalar applies to all thetas
    model.params.theta.set_val(0.7)
    # Or pass an array of shape model.params.theta.val_shape
    model.params.theta.set_val(np.array([[0.7, 0.5, 0.1]]))


Fixing subsets of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It may sometimes be desirable to fix the values of certain parameters.
The `fixed` attribute of `SepiaParam` is a boolean array of size `val_shape` (all `False` by default)::

    model.params.lamWOs.fixed = np.array([[True, False]])

Change prior distribution and prior parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there are only four distributions supported for priors: Normal, Gamma, Beta, and Uniform.
*Note*: this user interface will probably change to be more extendable and user-friendly.
After calling `setup_model`, we can modify priors as follows::

    prior_dist_name = 'Normal'
    prior_mu = 0.1
    prior_sd = 2.0
    prior_bounds = [0, 1]
    model.params.theta.prior = SepiaPrior(model.params.theta, dist=prior_dist_name, params=[prior_mu, prior_sd],
                                          bounds=prior_bounds)




Change MCMC step sizes or step types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can manually change MCMC step types or step sizes::

    model.params.theta.mcmc.stepType = 'Uniform'
    model.params.theta.mcmc.stepParam = np.array([[0.5, 0.1, 0.3]])


Automatic MCMC step size tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic step size tuning based on YADAS::

    model.tune_step_sizes(n_burn, n_levels)


Run MCMC or add more samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To do MCMC sampling, call `do_mcmc`::

    model.do_mcmc(500)

To append more samples to the current samples, you can call it again::

    model.do_mcmc(500) # Now has 1000 total samples


Saving MCMC chains periodically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will build some functions to handle this more smoothly, but for now you could do something like::

    import pickle
    for chunk in range(10):
        model.do_mcmc(500)
        with open('samples%d.pkl' % chunk, 'wb') as f:
            pickle.dump(model.get_samples(numsamples=500), f)



Extracting MCMC samples
-----------------------

To extract MCMC samples to a dictionary format::

    # Select a fixed set of samples
    model.get_samples(nburn=0, sampleset=np.arange(100), flat=True, includelogpost=True)

    # Select a fixed number of samples
    model.get_samples(nburn=0, numsamples=200, flat=True, includelogpost=True)

    # Discarding nburn samples
    model.get_samples(nburn=50, numsamples=200, flat=True, includelogpost=True)

    # Keep samples in array format rather than flattening along parameter dimensions
    model.get_samples(nburn=50, numsamples=200, flat=False)


Making predictions
------------------

To make predictions, use the :ref:`sepiapredict` class.
There are different types of predictions, and predictions can be made
in the model space (w, u, v) or output space (y), with or without standardization.

Emulator predictions
^^^^^^^^^^^^^^^^^^^^

To predict from the emulator (eta portion of model), first set up the `SepiaEmulatorPrediction` object::

    # Provide input settings to predict at
    x_pred = np.linspace(0,1,9).reshape((9,1))
    t_pred = np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
    pred_samples = model.get_samples(numsamples=10)
    pred = SepiaEmulatorPrediction(x_pred=x_pred, samples=pred_samples, model=model, t_pred=t_pred)

To get w::

    predw = pred.get_w()

To get y on the standardized scale::

    predystd = pred.get_y_standardized()

To get y on the native (original) scale::

    predystd = pred.get_y_native()


Full model predictions
^^^^^^^^^^^^^^^^^^^^^^

To predict from full model (including observation noise, and discrepancy, if applicable)::

    x_pred = np.linspace(0,1,9).reshape(9,1)
    pred_samples = model.get_samples(numsamples=7)
    pred = SepiaFullPrediction(x_pred, pred_samples, model)

To get u, v::

    predu, predv = pred.get_u_v()

To get discrepancy::

    preddstd = pred.get_discrepancy_standardized() # Standardized scale
    predd = pred.get_discrepancy_native()          # Native/original scale

To get y::

    predysimstd = pred.get_ysim_standardized() # Standardized scale
    predyobs=pred.get_yobs_native()            # Native/original scale

Cross-validation
^^^^^^^^^^^^^^^^

By default, leave-one-out cross validation is done on the emulator model::

    pred_samples = model.get_samples(numsamples=10)
    CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model)

You can also provide custom sets of indices to leave out in turn, such as leaving out chunks of 10 examples at a time,
and you can add residual variance to the predictions::

        leave_out_inds = np.array_split(np.arange(m), 5)
        pred_samples = model.get_samples(numsamples=7)
        CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, leave_out_inds=leave_out_inds, addResidVar=True)


Hierarchical or shared theta models
-----------------------------------

Coming soon


SepiaPlot visualization utilities
---------------------------------

Coming soon