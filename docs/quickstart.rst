.. _workflow:

General workflow guide
======================

The general workflow in SEPIA is summarized by:

    1. Instantiate :ref:`sepiadata` object with all data relevant to the problem.
    2. Use :ref:`sepiadata` methods to do data transformations/rescaling and create basis matrices for multivariate-output data.
    3. Create :ref:`sepiamodel` object using instantiated :ref:`sepiadata` object.
    4. Do MCMC to sample from the posterior distribution of the model parameters.
    5. Analyze the results: summarize posterior distributions, make predictions from the model, or perform sensitivity analysis.

The sections below give details on each step. We also include a section on more complex model types (hierarchical and shared theta models).

Data setup
----------

The first step is to set up a :ref:`sepiadata` object containing all of the data types that will be needed in the model.
Specifics of the model (whether or not the model is emulator-only, whether there is multivariate or univariate output,
whether or not there are controllable/experimental condition inputs) are inferred from the data structure, so it is
important to get it set up correctly. The data structure also handles various transformations and sets up basis
functions, so that users are not required to recreate these steps by hand. (That is, raw data can be passed in
without doing any transformations, and we recommend this so that downstream methods can untransform data as needed.)

The basic constructor call looks like::

    data = SepiaData(<inputs>)

The inputs given depend on the type of model and problem setup. Possible inputs are described in the table:

====================  =======================================================  =========================
   Possible inputs     Description                                              Shape
====================  =======================================================  =========================
x_sim                 Controllable simulation inputs.                           (n, p)
t_sim                 Other simulation inputs.                                  (n, q)
y_sim                 Simulation outputs.                                       (n, ell_sim)
y_ind_sim             Indices for multivariate sim outupts.                     (ell_sim, )
x_obs                 Controllable observed data inputs.                        (m, p)
y_obs                 Observation outputs.                                      (m, ell_obs)
y_ind_obs             Indices for multivariate observation outputs.             (ell_obs, )
x_cat_ind             List to indicate categorical x inputs.                    length p
t_cat_ind             List to indicate categorical t inputs.                    length q
xt_sim_sep            List of design matrices for Kronecker-separable design.   length depends on design
====================  =======================================================  =========================

In the table, `n` is the number of simulation runs, `m` is the number of observed data instances, and `ell` are the
multivariate output dimensions (if applicable). Unless indicated otherwise, all inputs are `numpy` arrays.

We emphasize that depending on the problem type, many of these inputs may not be used. For instance, if there is only
simulation data (an emulator-only model), none of `x_obs`, `y_obs`, or `y_ind_obs` will be used.
See :ref:`helpful-code-snippets` for examples of different types of data setup.

Notes:
    * For observed data, we also accept ragged observations in which the
      indices for the multivariate outputs differ for each observation. In this case, `y_obs` and `y_ind_obs` are given as
      lists (length `m`) of 1D `numpy` arrays.
    * For simulation-only (emulator) models, the distinction between `x` and `t` is not important, but it becomes important
      when observed data is included as only `t` variables will be calibrated (`x` are assumed known). Also note that
      for any model, if `x_sim` is not provided, a "dummy" `x` is set up with all values equal to 0.5. This does not affect
      the model and is generally not accessed by the user, but it facilitates unified treatment of different model types.
    * `xt_sim_sep` is only used in the special case of separable Kronecker-product input designs; it is a list of 2 or
      more design components that, through Kronecker expansion, produce the full input space (`x` and `t`) for the simulations.
    * The `SepiaData` constructor does some error-checking of the inputs, but it is still incumbent on the user to verify
      that the setup correctly reflects the problem of interest. Use `print(data)` on an instantiated `SepiaData` object
      to see printed information about the model that can be useful for checking.

Transformations
^^^^^^^^^^^^^^^

Transformations (standardization of `y`, rescaling of inputs to the unit cube) are important for the default priors
to work well on diverse data sets. After setting up the :ref:`sepiadata` object, users should always call the following
methods::

    data.transform_xt()
    data.standardize_y()

See :ref:`sepiadata` documentation for optional arguments used to customize the transformations.


Basis setup
^^^^^^^^^^^

For multivariate outputs, SEPIA uses basis functions to reduce the problem dimensionality. Basis function matrices must be
set up to represent the `y` values (done using principal components analysis, or PCA, on the simulation `y` values).
Optionally, a second set of basis functions may be set up to represent model discrepancy (systematic difference between simulation and observation data).

Basis matrices may be set up as follows::

    # PC basis
    data.create_K_basis(n_pc=5)     # With 5 PCs
    data.create_K_basis(n_pc=0.99)  # Enough PCs for at least 99 pct variance explained
    data.create_K_basis(K=K)        # Pass in custom K basis

    # Discrepancy basis -- optional
    data.create_D_basis(D_type='linear')  # Set up linear discrepancy
    data.create_D_basis(D=D)              # Pass in custom D basis

Internally, the projections onto the PCA `K` basis are referred to as `w` (simulation data) and `u` (observed data), while the
projections of the observed data onto the discrepancy `D` basis are referred to as `v`.

Checking your setup
^^^^^^^^^^^^^^^^^^^

To check that your data structure is set up correctly::

    print(data)

Also, for certain model types, the plotting methods in the :ref:`sepiadata` class may be helpful (see class documentation for options)::

    # Plot data - only for multivariate-output models with both simulation and observed outputs
    data.plot_data()
    # K basis functions - only for multivariate-output models
    data.plot_K_basis()
    # Histograms of projections of data onto K basis functions - only for multivariate-output models
    data.plot_K_weights()
    # Pairs plots of projections of data onto K basis functions - only for multivariate-output models
    data.plot_u_w_pairs()
    # Residuals after projection onto K basis - only for multivariate-output models
    data.plot_K_residuals()

Model setup
-----------

Once the data has been set up and checked, setting up the :ref:`sepiamodel` object is one line::

    model = SepiaModel(data)


MCMC
----

The inference on model parameters is done using MCMC sampling to approximate the posterior distribution of the model
parameters. The default model setup uses priors, initial values, and MCMC step sizes that have been selected to be
reasonable for a variety of scaled/transformed data. All of these are stored as object attributes and can be edited by the user as
needed.

Checking priors, start values, and MCMC tuning parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper functions in the :ref:`sepiamodel` class print out the default setup::

    model.print_prior_info()  # Print information about the priors
    model.print_value_info()  # Print information about the starting parameter values for MCMC
    model.print_mcmc_info()   # Print information about the MCMC step types and step sizes

A peek into the source code for the three print methods will show you how to access the attributes if you desire to modify them.

For example, to modify the start values directly, you can use::

    # Single scalar value: applies to all thetas
    model.params.theta.set_val(0.7)
    # Or: pass an array of shape model.params.theta.val_shape
    model.params.theta.set_val(np.array([[0.7, 0.5, 0.1]]))

Step size tuning
^^^^^^^^^^^^^^^^

Before doing MCMC, it maybe helpful to run an additional automatic step size tuning procedure,
meant to adjust the step sizes to achieve better acceptance rates::

    model.tune_step_sizes(n_burn, n_levels)

Note that automatic step size tuning is not guaranteed to produce good MCMC sampling, as it uses a heuristic and may be
affected by the number of levels chosen for each step parameter (`n_levels`) and the number of samples taken at each
level (`n_burn`). After MCMC sampling, we strongly recommend checking the output using trace plots and other diagnostics to ensure
automatic step size tuning has produced reasonable results.

Sampling
^^^^^^^^

Whether or not step size tuning has been done first, MCMC sampling is another one-liner::

    model.do_mcmc(nsamp)

To continue sampling (append more samples to existing samples), you can just call `do_mcmc()` again::

    model.do_mcmc(1000) # When finished, will have nsamp + 1000 total samples

To extract samples into a friendly dictionary format (see :ref:`sepiamodel` documentation for full options)::

    samples = model.get_samples()                       # Default: returns all samples
    samples = model.get_samples(effectivesamples=True)  # Returns a set of "effective samples"
    samples = model.get_samples(numsamples=100)         # Returns 100 evenly-spaced samples

When the model contains `theta`, the samples dictionary will contain both `theta` (in [0, 1])
and `theta_native` (untransformed to original scale), in addition to all other model parameters.

Saving samples
^^^^^^^^^^^^^^

To save a samples dictionary, you can pickle the samples dictionary::

    with open('mysamples.pkl', 'wb') as f:
        pickle.dump(samples, f)

Or you could save each array in the dictionary separately::

    import numpy as np
    for k in samples.keys():
        np.save('mysamples_%s.npy' % k, samples[k])

Save and restore model state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We do not recommend pickling the `SepiaModel` object itself  as any changes to the class definitions
or package namespace could lead to problems when you try to load the saved model in the future.

Instead, we offer methods that save important information from the model in a simple dictionary format and restore
this information into a new `SepiaModel` object. This requires you to create the new `SepiaModel` object using the same
data as the original model before restoring the saved information, but is otherwise automatic::

        # Set up original model and do MCMC
        model = SepiaModel(data)
        model.tune_step_sizes(50, 10)
        model.do_mcmc(100)

        # Save model info
        model.save_model_info(file_name='my_model_state')

        # Set up new model using same data (or a new SepiaData object constructed from same original inputs)
        new_model = SepiaModel(data)

        # Restore model info into the new model
        new_model.restore_model_info(file_name='my_model_state')

Diagnostics
^^^^^^^^^^^

After sampling, various diagnostics can be helpful for assessing whether the sampling was successful.
Most of the diagnostics are visual and are contained in the :ref:`sepiaplot` module.
The plotting functions return a `matplotlib` figure handle, but an optional `save` argument can provide a filename
to directly save the figure.

Trace plots of the MCMC samples are shown using::

    fig = mcmc_trace(samples)
    plt.show()

Summary statistics of the samples::

    ps = param_stats(samples) # returns pandas DataFrame
    print(ps)

Box plots of the GP lengthscale parameters::

    fig = rho_box_plots(model)
    plt.show()

The remaining plot functions only apply to models with `theta` variables (i.e., they do not produce output for emulator-only models).
The autocorrelation function (ACF) of the `theta` variables shows how correlated the MCMC samples are across the chain.
High correlation values for a large number of lags indicate that the chain is moving slowly through the space,
and that the effective sample size (ESS) could be much smaller than the actual number of samples. That is, if the
samples are highly correlated up to, say, ten lags, then adding ten more samples is not adding much new information about the parameter.
Plot the ACF and get a printout of the effective sample size using::

    fig = plot_acf(model, nlags=30)
    plt.show()

A pairs plot of the `theta` values is shown using::

    fig = theta_pairs(samples)
    plt.show()


Predictions
-----------

Aside from learning about the posterior distributions of the parameters, users may also be interested in making
predictions from the model. There are several types of predictions that can be made, depending on the type of model
and the goals of the user. All are handled by the :ref:`sepiapredict` class and make use of the MCMC samples from the model.

Emulator predictions
^^^^^^^^^^^^^^^^^^^^

Emulator predictions can be made whether the model is emulator-only or not. The emulator portion of the model is a
surrogate model that captures the relationship between simulation inputs and simulation outputs. Therefore, emulator
predictions can be interpreted as predictions of what the simulator would output at particular input settings.

The first step is to set up the prediction object, which requires supplying some subset of the MCMC samples as well as
both controllable and other simulation inputs where predictions are desired::

    # Provide input settings for which to get predictions
    x_pred = np.linspace(0,1,9).reshape((9,1))
    t_pred = np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
    # Extract a samples dictionary for which to get predictions
    pred_samples = model.get_samples(numsamples=10)
    # Set up prediction object
    pred = SepiaEmulatorPrediction(x_pred=x_pred, samples=pred_samples, model=model, t_pred=t_pred)

Note that by default, residual variance (from the nugget term) is not added; use argument `addResidVar=True` to add this.
Argument `storeMuSigma=True` will store the process mean and variance for each sample in addition to the realizations.

Once the prediction object is created, various types of predictions can be extracted. The first is to get predictions
of the `w` values (the weights for the PCA basis, used as a representation of the simulation outputs internally
in the model)::

    w_pred = pred.get_w()

More likely, users will want to get predictions that are transformed back to the original (native) output space::

    y_pred = pred.get_y()

Predictions in the standardized output space are also available::

    ystd_pred = pred.get_y(std=True)

If `SepiaEmulatorPrediction` was initialized with argument `storeMuSigma=True`, the posterior mean vector and sigma matrix
of the process for each sample are obtained by::

    mu_pred, sigma_pred = pred.get_mu_sigma()

Cross-validation predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is often of interest to obtain cross-validated predictions from the emulator. That is, instead of predicting at new
input values, we want to predict at the original simulation input values. However, simply predicting at the input values
used to train the model will give an unrealistically low estimate of the emulator error.
Cross-validation leaves out subsets of the input/training data in turn, then predicts at the inputs for the left out set
to better evaluate the error that would be observed at those input values if they were not actually part of the training data.

To set up the cross-validation prediction, we only need to provide samples from the MCMC::

    pred_samples = model.get_samples(numsamples=10)
    CV_pred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model)

This does leave-one-out cross-validation on the original simulation inputs.

Now the predictions can be compared to the original data to assess the error::

    CV_pred_y = CV_pred.get_y()
    residuals = CV_pred_y - model.data.sim_data.y

We can also customize the leave-out sets::

    leave_out_inds = np.array_split(np.arange(m), 5)
    pred_samples = model.get_samples(numsamples=7)
    CV_pred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, leave_out_inds=leave_out_inds)

Full predictions
^^^^^^^^^^^^^^^^

Full model predictions are slightly more complicated than emulator predictions because there are different options,
including whether we want multivariate predictions at the simulation or observed indices and whether we want to include
discrepancy (if applicable).

Set up the predictor instance::

    x_pred = np.linspace(0,1,9).reshape((9,1))
    t_pred = np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
    pred_samples = model.get_samples(numsamples=10)
    pred = SepiaFullPrediction(x_pred=x_pred, samples=pred_samples, model=model, t_pred=t_pred)

To extract predictions of the PCA projections `u` and discrepancy projections `v`::

    upred, vpred = pred.get_u_v()

To extract emulator-only predictions from the full model (not including discrepancy)::

    y_sim_pred = self.get_ysim(as_obs=False, std=False, obs_ref=0)

If `as_obs=False`, it will predict at the simulation data indices, otherwise at the observed data indices.
The argument `std` functions similarly to the emulator-only case: `std=False` returns predictions on the native space
while `std=True` returns them on the standardized space.
The `obs_ref` argument is used for cases where each observed data instance is ragged (has different multivariate indices),
to select which set of observation indices is used (only apples if `as_obs=True`).

To extract full model predictions (including discrepancy)::

    y_obs_pred = pred.get_yobs()

Note this function has the same optional arguments as `get_ysim`.

To extract just the predicted discrepancy::

    d_pred = pred.get_discrepancy()

Once again, this has the same optional arguments as `get_ysim`.

The posterior mean vector and sigma matrix of the process for each sample are obtained by::

    mu_pred, sigma_pred = pred.get_mu_sigma()


Sensitivity analysis
--------------------

Sensitivity analysis in SEPIA is based on `Sobol indices <https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_.

The syntax is::

    model.do_mcmc(1000)
    samples = model.get_samples(20)
    sens = sensitivity(model, samples)

For additional options, see :ref:`sepiasens`.

Hierarchical or shared theta models
-----------------------------------

Shared theta models are collections of models for which some of the thetas should be shared between the models.
This means the shared thetas will be sampled only once during MCMC across all the models, but that the likelihood
evaluation will take into account the likelihood from all the models.

Hierarchical theta models are collections of models for which some of the thetas are linked by a Normal hierarchical
model. In contrast to a shared theta model, this means that the thetas will differ between models, but when
being sampled during MCMC, they will be linked by a hierarchical specification, which typically induces "shrinkage" so
that the thetas tend to be more similar to each other than they would be if they were modeled as independent across models.

The syntax for both cases is similar. First, we set up each model, then put them in a list::

    m1 = SepiaModel(d1)
    m2 = SepiaModel(d2)
    m3 = SepiaModel(d3)
    model_list = [m1, m2, m3]

Then, we need to specify which thetas are shared or modeled hierarchically. The way to do this is with a numpy array
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

At this time, step size tuning is not implemented for shared or hierarchical models, but a reasonable approximation
might be to run step size tuning on each model separately before creating the shared/hierarchical model object.