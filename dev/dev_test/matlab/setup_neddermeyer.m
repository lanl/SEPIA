% sets up test case for comparing matlab neddermeyer to python neddermeyer

function res = setup_neddermeyer(seed,n_mcmc,sens,n_burn,n_lev)
    
    fprintf('\nStarting matlab setup_multi_sim_and_obs.m\n')

    addpath(genpath('GPMSAmatlab'))
    
    % Set up GPMSA
    dataStruct = neddeg()
    paramout = setupModel(dataStruct.obsData, dataStruct.simData);

    %% Do step size tuning
    rng(seed,'twister');
    if n_lev > 0
        step = stepsize(paramout, double(n_burn), double(n_lev));
        rhoUwidth = step.mcmc.rhoUwidth;
        lamUzwidth = step.mcmc.lamUzwidth;
        lamWswidth = step.mcmc.lamWswidth;
        lamWOswidth = step.mcmc.lamWOswidth;
    else
        rhoUwidth = [];
        lamUzwidth = [];
        lamWswidth = [];
        lamWOswidth = [];
    end

    %% Do mcmc samples
    rng(seed,'twister');
    if n_mcmc > 0
        mcmc = gpmmcmc(paramout, n_mcmc);
        mcmc_out.betaU = [mcmc.pvals.betaU]';
        mcmc_out.lamUz = [mcmc.pvals.lamUz]';
        mcmc_out.lamWs = [mcmc.pvals.lamWs]';
        mcmc_out.lamWOs = [mcmc.pvals.lamWOs]';
        mcmc_out.logPost = [mcmc.pvals.logPost]';
    else
        mcmc_out = [];
    end

    % Do sensitivity analysis
    % For now will just compare smePm, stePm to check against sepia
    if sens == 1
        sa = gSens(mcmc);
        smePm = sa.smePm;
        stePm = sa.stePm;
    else
        smePm = [];
        stePm = [];
    end

    res.smePm = smePm;
    res.stePm = stePm;
    res.mcmc = mcmc_out