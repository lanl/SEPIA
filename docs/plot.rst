.. _sepiaplot:

SepiaPlot
=========

SepiaPlot has visualization utilities which usually accept a SepiaData object.

.. automodule:: sepia.SepiaPlot
    :members:

    plot_data(data)               # Plot data
    plot_K_basis(data)            # Show K basis functions
    plot_K_weights(data)          # Show histograms of projections of data onto K basis functions
    plot_u_w_pairs(data)          # Show pairs plots of projections of data onto K basis functions
    plot_K_residuals(data)        # Show residuals after projection onto K basi
    theta_pairs(samples)          # Show pairs plots of thetas
    mcmc_trace(samples)           # Show trace plots of mcmc for all parameters
    param_stats(samples)          # Show summary statistics for mcmc
    rho_box_plots(model)          # Show boxplots of transformed GP lengthscale parameters
    plot_acf(model,nlags)         # Show autocorrelation function for thetas
    pca_projected_data(data)      # Compare data and PCA representations
    cv_pred_vs_true(model,cvpred) # Compare true and predicted PC weights from cross validation prediction object
