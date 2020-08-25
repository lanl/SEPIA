% Sets up test case for comparing univ sim and obs to python

function res = setup_univ_sim_and_obs(m, n, seed, n_lik, n_mcmc, n_pred)

    fprintf('\nStarting matlab setup_univ_sim_and_obs.m\n')

    addpath(genpath('GPMSAmatlab'))

    rng(seed,'twister');

    %% Sim data
    x = 0.5 * ones(m, 1);
    t = linspace(0, 1, m)';
    y = 10 .* x.^2 .* cos(10.*t);
    y_std = y - mean(y);
    y_sd = std(y_std);
    y_std = y_std/y_sd;
    
    %% Obs data
    sig_obs = 0.1;
    x_obs = 0.5 * ones(n, 1);
    t_obs = linspace(0, 1, n)';
    y_obs = 10 .* x_obs.^2 .* cos(10.*t_obs) + sig_obs * randn(n, 1);
    y_obs_std = (y_obs - mean(y))/y_sd;

    %% Make structure for model
    simData.x = [x, t]; 
    simData.yStd = y_std';
    simData.orig.y = y;
    simData.orig.ymean = mean(y);
    simData.orig.yd = y_sd;
    simData.Ksim = [1];
    data.simData = simData;
    
    for i=1:n
      obsData(i).x = [0.5];
      obsData(i).y = y_obs(i);
      obsData(i).Kobs = [1];
      obsData(i).Dobs = [];
      obsData(i).yStd = y_obs_std(i);
    end
    data.obsData = obsData;

    % Set up GPMSA
    paramout = setupModel(data.obsData, data.simData);
    
    %% Do likelihood calls
    ll = 0;
    ll_time = 0;
    if n_lik > 0
        fprintf('doing matlab likelihood calls\n')
        C.var='all';
        ll = computeLogLik(paramout.model,paramout.data,C).logLik;
        tic;
        for i = 1:n_lik
            computeLogLik(paramout.model,paramout.data,C);
        end
        ll_time = toc;
    end

    rng(seed,'twister');
    %% Do mcmc samples
    if n_mcmc > 0
        tic;
        mcmc = gpmmcmc(paramout, n_mcmc);
        mcmc_time = toc;
        mcmc_out.theta = [mcmc.pvals.theta]';
        mcmc_out.betaU = [mcmc.pvals.betaU]';
        mcmc_out.lamUz = [mcmc.pvals.lamUz]';
        mcmc_out.lamWs = [mcmc.pvals.lamWs]';
        mcmc_out.lamWOs = [mcmc.pvals.lamWOs]';
        mcmc_out.lamOs = [mcmc.pvals.lamOs]';
        mcmc_out.logPost = [mcmc.pvals.logPost]';
    else
        mcmc_time = 0;
        mcmc_out = [];
    end

    if n_pred > 0
        %% make some predictions to test
        % basic samples prediction
        rng(seed,'twister');
        mcmc.model.debugRands=true;
        pred=gPredict([0.5],mcmc.pvals(1:n_pred),mcmc.model,mcmc.data,'returnMuSigma',1,'mode','wpred');
        pred_w=pred.w;
        pred_Myhat=pred.Myhat;
        pred_Syhat=[pred.Syhat{:}];
    else
        pred_w = [];
        pred_Myhat = [];
        pred_Syhat = [];
        pred2_u = [];
        pred2_Myhat = [];
        pred2_Syhat = [];
    end
    
    % Store stuff needed for python in a struct
    res.y = simData.orig.y;
    res.xt = simData.x;
    res.y_obs = y_obs;
    res.x_obs = 0.5 * ones(n, 1);
    res.ll = ll;
    res.ll_time = ll_time;
    res.mcmc = mcmc_out;
    res.mcmc_time = mcmc_time;
    res.pred_w = pred_w;
    res.pred_Myhat = pred_Myhat;
    res.pred_Syhat = pred_Syhat;
end