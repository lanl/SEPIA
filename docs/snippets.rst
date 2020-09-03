.. _helpful-code-snippets:

Helpful Code Snippets
=====================

Unlike notebooks, these are not self-contained examples, but are meant to be a quick reference for specific tasks.
These are not necessarily exhaustive examples; see full class documentation for all possible arguments and options.
For a more full walkthrough, see :ref:`workflow`.

:ref:`SepiaData inputs`

:ref:`SepiaData operations`

:ref:`Model setup`

:ref:`Customize and run MCMC`

:ref:`Extract MCMC samples`

:ref:`Make predictions`

:ref:`Hierarchical or shared theta models`


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
* `x_cat_ind`: list to identify columns of `x` that are categorical variables, where 0 means not categorical and an integer gives the number of categories. Categorical variables should be nonzero integers.
* `t_cat_ind`: list to identify columns of `t` that are categorical variables, similar to `x_cat_ind`.


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
    # Indicate that third column of x is categorical with 5 categories (takes values in [1, 2, 3, 4, 5])
    data = SepiaData(x_sim=x, y_sim=y, x_cat_ind=[0, 0, 5])

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

Multivariate-output simulation and observed data with ragged observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ragged observations means the observed data indices vary across observation data instances.
In this case, `y_obs` and `y_ind_obs` are now lists instead of numpy arrays::

    y_obs = [np.array([[0.3, 0.5, 0.7]]), np.array([[0.1, 0.4, 0.6, 0.9]])
    y_ind_obs = [np.array([1, 2, 3]), np.array([0.5, 2.5, 4, 6])]
    data = SepiaData(t_sim=t, y_sim=y, y_ind_sim=y_ind, y_obs=y_obs, y_ind_obs=y_ind_obs) # No controllable input

SepiaData operations
--------------------

Regardless of the inputs given to `SepiaData`, there are a few key methods which generally should be called before
setting up the model.

Transformations
^^^^^^^^^^^^^^^

First, we want to transform `x` and `t` inputs to the unit hypercube::

    data.transform_xt()

Next, we want to standardize the `y` outputs::

    data.standardize_y()

Basis functions
^^^^^^^^^^^^^^^

If the outputs are multivariate, we want to set up a principal component (PC) basis and optionally, a discrepancy basis::

    # PC basis
    data.create_K_basis(n_pc=5)     # With 5 PCs
    data.create_K_basis(n_pc=0.99)  # Enough PCs for at least 99 pct variance explained

    # Discrepancy basis -- optional
    data.create_D_basis(D_type='linear')  # Default linear discrepancy
    data.create_D_basis(D=D)              # Pass in custom D basis

Plotting
^^^^^^^^

Plot data and basis function diagnostics (some model types are not covered by these functions)::

    data.plot_data()        # Plot data
    data.plot_K_basis()     # Show K basis functions
    data.plot_K_weights()   # Show histograms of projections of data onto K basis functions
    data.plot_u_w_pairs()   # Show pairs plots of projections of data onto K basis functions
    data.plot_K_residuals() # Show residuals after projection onto K basis


Set up model
------------

Once the data structure is set up correctly, the inputs are in the unit hypercube, the outputs are standardized,
and basis vectors are created (for multivariate output), we are ready to set up the Sepia model::

    model = SepiaModel(data)

The model parses the `SepiaData` structure to understand what kind of model is being set up and does a lot of
precomputation of various quantities to prepare for likelihood evaluations.
It also sets up default priors, MCMC step types and step sizes, and default starting values for MCMC.

To see information about the default setup, you can use::

    model.print_prior_info()
    model.print_value_info()
    model.print_mcmc_info()


Customize and run MCMC
----------------------

After instantiating the `SepiaModel` object, various aspects of the MCMC can be customized prior to doing MCMC.

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
After instantiating the `SepiaModel`, we can modify priors as follows::

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

Automatic step size tuning based on `YADAS`_::

    model.tune_step_sizes(n_burn, n_levels)

.. _YADAS: https://arxiv.org/abs/1103.5986

MAP optimization for start values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Numerical optimization of the log likelihood will reset start values to the best points found.
:ref:`sepiaoptim` implements two gradient-free optimizers::

    optimizer = SepiaOptim(model)
    nm_opt_result = optimizer.nelder_mead(log_transform=['betaU','betaV','lamVz','lamWs','lamWOs','lamOs'])
    nm_opt_param = nm_opt_result[2]          # Optimized params, untransformed
    optimizer.set_model_params(nm_opt_param) # Sets into model parameter values
    pso_opt_result = optimizer.particle_swarm(log_transform=['betaU','betaV','lamVz','lamWs','lamWOs','lamOs'])
    pso_opt_param = pso_opt_result[5]
    optimizer.set_model_params(pso_opt_param)

Run MCMC or add more samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To do MCMC sampling, call `do_mcmc`::

    model.do_mcmc(500)

To append more samples to the current samples, you can call it again::

    model.do_mcmc(500) # Now has 1000 total samples


Saving MCMC chains or model info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will build some functions to handle this more smoothly, but for now you could do something like::

    import pickle
    for chunk in range(10):
        model.do_mcmc(500)
        with open('samples%d.pkl' % chunk, 'wb') as f:
            pickle.dump(model.get_samples(numsamples=500), f)

See also `SepiaModel.save_model_info()` and `SepiaModel.restore_model_info()`.

Extract MCMC samples
--------------------

To extract MCMC samples to a dictionary format::

    # Select a fixed set of samples
    model.get_samples(nburn=0, sampleset=np.arange(100), flat=True, includelogpost=True)

    # Select a fixed number of samples
    model.get_samples(nburn=0, numsamples=200, flat=True, includelogpost=True)

    # Discarding nburn samples
    model.get_samples(nburn=50, numsamples=200, flat=True, includelogpost=True)

    # Keep samples in array format rather than flattening along parameter dimensions
    model.get_samples(nburn=50, numsamples=200, flat=False)

    # Returns only a set of "effective samples" determined by effective sample size
    samples = model.get_samples(effectivesamples=True)

MCMC diagnostics
----------------

Several graphical diagnostics are available::

    plot_acf(model, nlags=30) # Autocorrelation function and effective sample size calculation
    mcmc_trace(samples)       # Trace plots
    theta_pairs(samples)      # Pairs plots of theta variables
    rho_box_plots(model)      # Box plots of GP lengthscale parameters
    param_stats(samples)      # Summary statistics of parameters

Each returns a `matplotlib` figure object that you can save using `plt.savefig()` or show using `plt.show()`.

Make predictions
----------------

To make predictions, use the :ref:`sepiapredict` class.
There are different types of predictions, and predictions can be made
in the model space (`w, u, v`) or output space (`y`), with or without standardization.

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

    predystd = pred.get_y(std=True)

To get y on the native (original) scale::

    predystd = pred.get_y()


Full model predictions
^^^^^^^^^^^^^^^^^^^^^^

To predict from full model (including observation noise, and discrepancy, if applicable)::

    x_pred = np.linspace(0,1,9).reshape(9,1)
    pred_samples = model.get_samples(numsamples=7)
    pred = SepiaFullPrediction(x_pred, pred_samples, model)

To get u, v::

    u_pred, v_pred = pred.get_u_v()

To get discrepancy::

    preddstd = pred.get_discrepancy(std=True) # Standardized scale
    predd = pred.get_discrepancy()            # Native/original scale

To get simulated y::

    predysimstd = pred.get_ysim(std=True) # Standardized scale
    predysim = pred.get_ysim()            # Native/original scale

To get y (simulator+discrepancy)::

    predy = pred.get_yobs()                            # Native/original scale
    predystd = pred.get_yobs(std=True)                 # Standardized scale
    predystdobs = pred.get_yobs(std=True, as_obs=True) # Standardized scale at only observed data locations x_obs
    
Cross-validation
^^^^^^^^^^^^^^^^

By default, leave-one-out cross validation is done on the emulator model::

    pred_samples = model.get_samples(numsamples=10)
    CV_pred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model)

    CV_pred_y = CV_pred.get_y()

You can also provide custom sets of indices to leave out in turn, such as leaving out chunks of 10 examples at a time,
and you can add residual variance to the predictions::

    leave_out_inds = np.array_split(np.arange(m), 5)
    pred_samples = model.get_samples(numsamples=7)
    CV_pred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, leave_out_inds=leave_out_inds, addResidVar=True)


Hierarchical or shared theta models
-----------------------------------

The syntax for both cases is similar. First, set up each model, then put them in a list::

    m1 = SepiaModel(d1)
    m2 = SepiaModel(d2)
    m3 = SepiaModel(d3)
    model_list = [m1, m2, m3]

Then we need to specify which thetas are shared or modeled hierarchically. The way to do this is with a numpy array
of size `(j, n_models)` where each row represents one of the shared/hierarchical theta variables,
and each column gives the index of the shared/hierarchical theta in the respective model. For instance::

    theta_inds = np.array([[0, 0, 0], [1, 1, 2], [-1, 2, 3]])

This means that the first shared/hierarchical theta is `theta_0` in model 1, `theta_0` in model 2, and `theta_0` in model 3.
The second shared/hierarchical theta is `theta_1` in model 1, `theta_1` in model 2, and `theta_2` in model 3.
The third shared/hierarchical theta is *not* in model 1, is `theta_2` in model 2, and is `theta_3` in model 3.
The index -1 is used to indicate that a particular shared/hierarchical theta is not in a particular model.

Then the model setup is::

    shared_model = SepiaSharedThetaModels(model_list, theta_inds)     # Shared version
    hier_model = SepiaHierarchicalThetaModels(model_list, theta_inds) # Hierarchical version

MCMC is done similarly to regular models::

    shared_model.do_mcmc()

Step size tuning is not supported on shared or hierarchical theta models.