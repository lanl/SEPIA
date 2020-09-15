.. _sepia-for-gpmsa-users:

SEPIA for GPMSA users
=====================

SEPIA implements essentially the same core methodology as GPMSA, and we have verified that we get the same results using
SEPIA as we do for GPMSA for several synthetic data sets.
However, the user interface for setting up the model has some important differences.

In GPMSA, all data and input transformations and basis set up are done manually by the user and stored into a struct.
This requires not only applying manual transformations and basis creation but also interpolation in the case that
the simulation data is measured on a different set of multivariate indices than the observed data.
One of the key differences between GPMSA and SEPIA lies in the data set up and handling.
As an example, we'll compare code in GPMSA for setting up a simulator-only model to the equivalent SEPIA code.

Example GPMSA code::

    % Transform t to unit hypercube, assuming t is set of calibration inputs (matrix m by q)
    tmin = min(t);
    tmax = max(t);
    trange = tmax - tmin;
    t = bsxfun(@minus, t, tmin);
    t = bsxfun(@rdivide, t, trange);

    % Add dummy variable for controllable inputs
    m = size(t, 1);
    xt = [0.5*ones(m,1), t];

    % Standardize simulations, where ysim is set of simulation outputs (matrix m by ell_sim)
    % and ysimind is set of indices (ell_sim, 1)
    ysimmean = mean(ysim, 2);
    ysimStd = bsxfun(@minus, ysim, ysimmean);
    ysimsd = std(ysimStd(:));
    ysimStd = ysimStd / ysimsd;

    % Create simulation K basis using PCA
    [U, S, ~] = svd(ysimStd, 0);
    pu = 11;
    Ksim = U(:,1:pu) * S(1:pu,1:pu) / sqrt(m);

    % Create simData struct
    simData.Ksim = Ksim;
    simData.yStd = ysimStd;
    simData.x = xt;
    simData.orig.y = ysim;
    simData.orig.ymean = ysimmean;
    simData.orig.ysd = ysimsd;
    simData.time = ysimind;

    % Create model
    model = setupModel([], simData);

Example SEPIA code::

    # Create SepiaData object (automatically sets up dummy x input since x_sim not passed in)
    data = SepiaData(t_sim=t, y_sim=ysim, y_ind_sim=ysimind)

    # Transform inputs to unit hypercube
    data.transform_xt()

    # Standardize y
    data.standardize_y()

    # Create K basis
    data.create_K_basis(pu)

    # Set up model
    model = SepiaModel(data)

As you can see, the SEPIA code is much more concise and handles the standardization, basis creation, and setting of
object properties for the user.
If `obs_data` were also present, the calls to `data.transform_xt()` and `data.standardize_y()` would automatically apply
the same transformation used for the simulation data to the observed data.
In addition, `data.create_K_basis()` would automatically set up a basis for the observed data, interpolating to different
output indices if needed.

The major differences end there; functions in GPMSA have direct analogues as functions or methods in SEPIA.
Here is a quick reference for translating to SEPIA from GPMSA.

====================  ===============================  ===========================================================================
   GPMSA function      SEPIA method/function             Key differences
====================  ===============================  ===========================================================================
 setupModel             SepiaModel(data)                Create SepiaData object first, do transformations/basis, then SepiaModel.
 gpmmcmc                model.do_mcmc()
 gPredict               SepiaPredict object methods     Instantiate SepiaPredict object, then extract items of interest
 gXval                  Part of SepiaPredict
 gSens                  sensitivity(model, samples)
====================  ===============================  ===========================================================================