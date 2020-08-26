% Sets up test case for comparing univ sim only to python

function res = setup_univ_sim_only(m, seed, n_lik, n_mcmc, n_pred, n_lev, n_burn)

    fprintf('\nStarting matlab setup_univ_sim_only.m\n')

    addpath(genpath('GPMSAmatlab'))

    rng(seed,'twister');

    % x and t
    x = 0.5 * ones(m, 1);
    t = linspace(0, 1, m)';
    y = 10 .* x.^2 .* cos(10.*t);
    % x only
    %x = linspace(0, 1, m)';
    %y = 10 .* x.^2 .* cos(10.*x);
    y_std = y - mean(y);

    y_sd = std(y_std);
    y_std = y_std/y_sd;

    %% Make structure for model
    % x and t
    simData.x = [x, t]; 
    % x only
    %simData.x = x; 
    simData.yStd = y_std';
    simData.orig.y = y;
    simData.orig.ymean = mean(y);
    simData.orig.yd = y_sd;
    simData.Ksim = [1];
    data.simData = simData;

    % Set up GPMSA
    paramout = setupModel([], data.simData);
    
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
        tic;
        mcmc = gpmmcmc(paramout, n_mcmc);
        mcmc_time = toc;
        mcmc_out.betaU = [mcmc.pvals.betaU]';
        mcmc_out.lamUz = [mcmc.pvals.lamUz]';
        mcmc_out.lamWs = [mcmc.pvals.lamWs]';
        mcmc_out.lamWOs = [mcmc.pvals.lamWOs]';
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
        pred=gPredict([0.5,0.5],mcmc.pvals(1:n_pred),mcmc.model,mcmc.data,'returnMuSigma',1);
        pred_w=pred.w;
        pred_Myhat=pred.Myhat;
        pred_Syhat=[pred.Syhat{:}];
        % suitable for plotting
        rng(seed,'twister')
        mcmc.model.debugRands=true;
        pred_plot_xpred= [0.5*ones(10,1) linspace(0,1,10)'];
        pred_plot=gPredict(pred_plot_xpred,mcmc.pvals(100:100:1000),paramout.model,paramout.data);
        pred_plot_w=squeeze(pred_plot.w);
    else
        pred_w = [];
        pred_Myhat = [];
        pred_Syhat = [];
        pred_plot_w = [];
    end
    
    % Store stuff needed for python in a struct
    res.y = simData.orig.y;
    res.xt = simData.x;
    res.ll = ll;
    res.ll_time = ll_time;
    res.rhoUwidth = rhoUwidth;
    res.lamUzwidth = lamUzwidth;
    res.lamWswidth = lamWswidth;
    res.lamWOswidth = lamWOswidth;
    res.mcmc = mcmc_out;
    res.mcmc_time = mcmc_time;
    res.pred_w = pred_w;
    res.pred_Myhat = pred_Myhat;
    res.pred_Syhat = pred_Syhat;
    res.pred_plot_w = pred_plot_w;
end