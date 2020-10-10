Optimization for start values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The start values for MCMC are shown by the `model.print_value_info()` method and may be modified directly if needed.
Step size tuning will also reset the start values based on the samples collected during step size tuning, and will
hopefully start the sampling in a higher-posterior region than the default start values.

If desired, you can also try to optimize the log posterior to get point estimates of the parameters which could be
even better start values. `SepiaOptim` implements two gradient-free optimizers: Nelder-Mead and particle swarm.

The first step is to instantiate a `SepiaOptim` object using the model::

    optimizer = SepiaOptim(model)

Then call one of the optimization routines; here is Nelder-Mead::

    nm_opt_result = optimizer.nelder_mead(log_transform=['betaU','betaV','lamVz','lamWs','lamWOs','lamOs'])

The `log_transform` argument lists variables that should have a log transform applied inside the optimizer; this
generally applies to parameters that should be positive (such as betas and lams).

You can then inspect the optimized values, and if desired, set them into the model using::

    nm_opt_param = nm_opt_result[2]          # Optimized params, untransformed
    optimizer.set_model_params(nm_opt_param) # Sets into model parameter values

Particle swarm works similarly::

    pso_opt_result = optimizer.particle_swarm(log_transform=['betaU','betaV','lamVz','lamWs','lamWOs','lamOs'])
    pso_opt_param = pso_opt_result[5]
    optimizer.set_model_params(pso_opt_param)

The optimizers have many other options; see :ref:`sepiaoptim` for details.