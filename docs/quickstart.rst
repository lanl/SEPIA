.. _workflow:

General workflow guide
======================

Data setup
----------

The first step is to set up a :ref:`sepiadata` object containing all of the data types that will be needed in the model.
Specifics of the model (whether or not emulator-only, whether multivariate or univariate output, whether or not
there are controllable/experimental condition inputs) are inferred from the data structure, so it is
important to get it set up correctly. The data structure also handles various transformations and sets up basis
functions, so that users are not required to recreate these steps by hand. (That is, raw data can be passed in
without doing any transformations, and we recommend this so that downstream methods can handle untransforming data.)

====================  ================================================  =================
   Possible inputs     Description                                       Shape
====================  ================================================  =================
x_sim                 Controllable simulation inputs.                   (n, p)
t_sim                 Other simulation inputs.                          (n, q)
y_sim                 Simulation outputs.                               (n, ell_sim)
y_ind_sim             Indices for multivariate sim outupts.             (ell_sim, )
x_obs                 Controllable observed data inputs.                (m, p)
y_obs                 Observation outputs.                              (m, ell_obs)
y_ind_obs             Indices for multivariate observation outputs.     (ell_obs, )
====================  ================================================  =================

In the table, `n` is the number of simulation runs, `m` is the number of observed data instances, and `ell` are the
multivariate output dimensions (if applicable). Note that for observed data, we also accept ragged observations,
where `y_obs` and `y_ind_obs` are given as lists (length `m`) of arrays.

Transformations
^^^^^^^^^^^^^^^

Transformations (standardization of `y`, rescaling of inputs to the unit cube) are important for the default priors
to work well on diverse data sets. After setting up the :ref:`sepiadata` object, users should always call the following
methods::

    data.transform_xt()
    data.standardize_y()

See :ref:`sepiadata` documentation for optional arguments, though the defaults should generally work well.


Basis setup
^^^^^^^^^^^

For multivariate outputs, Sepia uses basis functions to reduce the problem dimensionality. Basis functions must be
set up to represent the `y` values (done by principal components analysis, or PCA), and optionally, a second set of basis
functions may be set up to represent model discrepancy (systematic difference between simulation and observation data).

These are set up as follows::

    # PC basis
    data.create_K_basis(n_pc=5)     # With 5 PCs
    data.create_K_basis(n_pc=0.99)  # Enough PCs for at least 99 pct variance explained
    data.create_K_basis(K=K)        # Pass in custom K basis

    # Discrepancy basis -- optional
    data.create_D_basis(type='linear')  # Default linear discrepancy
    data.create_D_basis(D=D)            # Pass in custom D basis

Internally, the projections onto the PCA `K` basis are referred to as `w` (simulations) and `u` (observed), while the
projections of the observed data onto the discrepancy `D` basis are referred to as `v`.

Checking your setup
^^^^^^^^^^^^^^^^^^^

To check that your data structure is set up correctly::

    print(data)

Also, use plotting methods in the :ref:`sepiadata` class to visualize the data (see class documentation for options)::

    data.plot_data()        # Plot data
    data.plot_K_basis()     # Show K basis functions
    data.plot_K_weights()   # Show histograms of projections of data onto K basis functions
    data.plot_u_w_pairs()   # Show pairs plots of projections of data onto K basis functions
    data.plot_K_residuals() # Show residuals after projection onto K basis

Model setup
-----------

Once the data has been set up and checked, setting up the :ref:`sepiamodel` object is one line::

    model = setup_model(data)


MCMC
----

The inference on model parameters is done using MCMC sampling to approximate the posterior distribution of the model
parameters. The default model setup uses priors, initial values, and MCMC step sizes that have been selected to be
reasonable for scaled/transformed data. All of these are stored as object attributes and can be edited by the user if
needed.

Helper functions in the :ref:`sepiamodel` class print out the default setup::

    model.print_prior_info()  # Print information about the priors
    model.print_value_info()  # Print information about the starting parameter values for MCMC
    model.print_mcmc_info()   # Print information about the MCMC step types and step sizes for each parameter

A peek into the code for the three print methods will show you how to access the attributes if you desire to modify them.

For example, to modify the start values directly, you can use::

    # Single scalar applies to all thetas
    model.params.theta.set_val(0.7)
    # Or pass an array of shape model.params.theta.val_shape
    model.params.theta.set_val(np.array([[0.7, 0.5, 0.1]]))

Step size tuning
^^^^^^^^^^^^^^^^

Before doing MCMC, it maybe helpful to run an additional automatic step size tuning procedure,
meant to adjust the step sizes to achieve better acceptance rates::

    model.tune_step_sizes(n_burn, n_levels)

Note that automatic step size tuning is not guaranteed to produce good MCMC sampling, as it uses a heuristic and may be
affected by the number of levels chosen for each step parameter (`n_levels`) and the number of samples taken at each
level (`n_burn`). After MCMC sampling, we strongly recommend checking the output using trace plots or other diagnostics to ensure
automatic step size tuning has produced reasonable results.

MAP optimization for start values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The start values for MCMC are shown by the `model.print_value_info()` method and may be modified directly if needed.
Step size tuning will also reset the start values based on the samples collected during step size tuning, and will
hopefully start the sampling in a higher-posterior region than the default start values.

If desired, you can also try to optimize the log posterior to get point estimates of the parameters which could be
even better start values::

    opt_prm = model.optim_logPost()

This method returns the optimized parameters and also sets the start values within the model object to these values.
Note that the values are found by numerical optimization and are not guaranteed to be the actual MAP values.

Sampling
^^^^^^^^

Whether or not step size tuning or MAP optimization has been done first, MCMC sampling is another one-liner::

    model.do_mcmc(nsamp)

To continue sampling (append more samples), you can just call `do_mcmc()` again::

    model.do_mcmc(1000) # When finished, will have nsamp + 1000 total samples

Diagnostics
^^^^^^^^^^^

After sampling, various diagnostics can be helpful for assessing whether the sampling was successful.
Most of the diagnostics are visual and are contained in the :ref:`sepiaplot` module.

The autocorrelation function (ACF) of the `theta` variables shows how correlated the MCMC samples are across the chain.
High correlation values for a large number of lags indicate that the chain is moving slowly through the space,
and that the effective sample size (ESS) could be much smaller than the actual number of samples. That is, if the
samples are highly correlated up to, say, ten lags, then adding ten more samples is not adding much new information about the parameter.
Plot the ACF and get a printout of the effective sample size using::

    plot_acf(model, nlags=30)

Some of the diagnostic methods take a samples dictionary as an argument, which you can extract from the model::

    samples = model.get_samples()

Then you can investigate trace plots and pairs plots of the `theta` variables::

    mcmc_trace(samples)
    theta_pairs(samples)

Summary statistics of the samples::

    param_stats(samples)

Box plots of the GP lengthscale parameters::

    rho_box_plots(model)


Predictions
-----------

Aside from learning about the posterior distributions of the parameters, users may also be interested in making
predictions from the model. There are several types of predictions that can be made, depending on the type of model
and the goals of the user. All are handled by the :ref:`sepiapredict` class and make use of the MCMC samples in the model.

    .. note:: The predictions class is still being finalized, so this section is subject to change.

Emulator predictions
^^^^^^^^^^^^^^^^^^^^

Emulator predictions can be made whether the model is emulator-only or not. The emulator portion of the model is a
surrogate model that captures the relationship between simulation inputs and simulation outputs. Therefore, emulator
predictions can be interpreted as predictions of what the simulator would output, and we're usually interested in
seeing the emulator output at input settings that were not included in the original simulation data set.

The first step is to set up the prediction object, which requires supplying some subset of the MCMC samples as well as
both controllable and other simulation inputs (the inputs where predictions are desired)::

    # Provide input settings to predict at
    x_pred = np.linspace(0,1,9).reshape((9,1))
    t_pred = np.tile(np.array([1,0,1]).reshape(1,3),(9,1))
    pred_samples = model.get_samples(numsamples=10)
    pred = SepiaEmulatorPrediction(x_pred=x_pred, samples=pred_samples, model=model, t_pred=t_pred)

Note that by default, residual variance (from the nugget term) is not added; use argument `addResidVar=True` to add this.

Once the prediction object is created, various types of predictions can be extracted. The first is to get predictions
of the `w` values (the weights for the PCA basis, used as a representation of the simulation outputs internally
in the model, but not necessarily as interpretable as the other types of predictions)::

    predw = pred.get_w()

More likely, users will want to get predictions that are transformed back to the original (native) output space::

    predystd = pred.get_y()

Predictions in the standardized output space are also available::

    predystd = pred.get_y(std=True)

Cross-validation predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is often of interest to obtain cross-validated predictions from the emulator. That is, instead of predicting at new
input values, we want to predict at the original simulation input values. However, simply predicting at the input values
used to train the model will give an unrealistically low level of error. Cross-validation leaves out subsets of the
input/training data in turn, predicting at the inputs for the left out set, to better evaluate the error that would be
observed at those input values if they were not actually part of the training data.

To set up the cross-validation prediction, we only need to provide samples from the MCMC::

    pred_samples = model.get_samples(numsamples=10)
    CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model)

This does leave-one-out cross-validation on the original simulation inputs.

Now the predictions can be compared to the original data to assess the error::

    CV_pred_y = CVpred.get_y()
    residuals = CV_pred_y - model.data.sim_data.y

We can also customize the leave-out sets::

    leave_out_inds = np.array_split(np.arange(m), 5)
    pred_samples = model.get_samples(numsamples=7)
    CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, leave_out_inds=leave_out_inds)



Full predictions
^^^^^^^^^^^^^^^^

Coming soon