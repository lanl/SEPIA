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

See :ref:`sepiadata` documentation for optional arguments, though the defaults will generally work well.


Basis setup
^^^^^^^^^^^

For multivariate outputs, Sepia uses basis functions to reduce the problem dimensionality. Basis functions must be
set up to represent the `y` values (done by principal components analysis), and optionally, a second set of basis
functions may be set up to represent model discrepancy (systematic difference between simulation and observation data).

These are set up as follows::

    # PC basis
    data.create_K_basis(n_pc=5)     # With 5 PCs
    data.create_K_basis(n_pc=0.99)  # Enough PCs for at least 99 pct variance explained

    # Discrepancy basis -- optional
    data.create_D_basis(type='linear')  # Default linear discrepancy
    data.create_D_basis(D=D)            # Pass in custom D basis

Checking your setup
^^^^^^^^^^^^^^^^^^^

To check that your data structure is set up correctly::

    print(data)

We are also working on plotting methods to show the data and the basis functions and projections. These will be
helpful for checking and for diagnosing potential problems prior to running the model. (TODO: coming soon)


Model setup
-----------

Once the data has been set up as shown above, setting up the :ref:`sepiamodel` object is easy::

    model = setup_model(data)


MCMC
----

The inference is done using MCMC sampling to approximate the posterior distribution of the model parameters. The
default model setup uses priors, initial values, and MCMC step sizes that have been selected to be reasonable for
scaled/transformed data. All of these are stored as object attributes and can be edited by the user if needed.

Step size tuning
^^^^^^^^^^^^^^^^

Before doing MCMC, it maybe helpful to run an additional automatic step size tuning procedure,
meant to adjust the step sizes to achieve better acceptance rates::

    model.tune_step_sizes(n_burn, n_levels)

Note that automatic step size tuning is not guaranteed to produce good MCMC sampling, as it uses a heuristic and may be
affected by the number of levels chosen for each step parameter (`n_levels`) and the number of samples taken at each
level (`n_burn`). We still strongly recommend checking the output using trace plots or other diagnostics to ensure
automatic step size tuning has produced reasonable results.

Sampling
^^^^^^^^

Whether or not step size tuning has been done, MCMC sampling is done using::

    model.do_mcmc(namp)


Diagnostics
^^^^^^^^^^^

TODO: coming soon


Predictions
-----------

Aside from learning about the posterior distributions of the parameters, users may also be interested in making
predictions from the model. There are several types of predictions that can be made, depending on the type of model
and the goals of the user. All are handled by the :ref:`sepiapredict` class and make us of the stored MCMC samples.

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

    predystd = pred.get_y_native()

Predictions in the standardized output space are also available::

    predystd = pred.get_y_standardized()

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

    CV_pred_y = CVpred.get_y_native()
    residuals = CV_pred_y - model.data.sim_data.y

We can also customize the leave-out sets::

    leave_out_inds = np.array_split(np.arange(m), 5)
    pred_samples = model.get_samples(numsamples=7)
    CVpred = SepiaXvalEmulatorPrediction(samples=pred_samples, model=model, leave_out_inds=leave_out_inds)



Full predictions
^^^^^^^^^^^^^^^^

Coming soon