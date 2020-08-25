% Sets up test case for comparing multi sim only to python

function res = setup_multi_sim_only(m, nt, nx, n_pc, seed, n_lik, n_mcmc, n_pred)

    fprintf('\nStarting matlab setup_multi_sim_only.m\n')

    addpath(genpath('GPMSAmatlab'))

    rng(seed,'twister');
    % Generate sim data using a Fourier basis to match n_pc
    x = repmat(linspace(0, 1, m), nx, 1)';
    beta = abs(0.5 * rand(nx, 1)) + 0.05;
    %beta = 1./len_scl;
    cov = zeros(m, m);
    for i = 1:m
        for j = 1:m
            cov(i, j) = exp(-sum(beta' .* (x(i, :) - x(j, :)).^2, 2));
        end
    end
    
    chCov = chol(cov + 1e-3 * eye(m));
    wt_gen = chCov' * randn(m, n_pc) .* 1./sqrt(double(cumsum(1:n_pc)));
    
    y = zeros(m, nt);
    time = linspace(0, 1, nt);
    for i = 1:n_pc
        for j = 1:m
            y(j, :) = y(j, :) + wt_gen(j, i) * cos(pi * double(i) * time);
        end
    end
    y = y';

    %% Make structure for model
    ymean = mean(y, 2);
    ysimStd = bsxfun(@minus, y, ymean);
    ysimsd = std(ysimStd(:));
    ysimStd = ysimStd/ysimsd;
    y_ind = time;
    [U, S, ~] = svd(ysimStd, 0);
    Ksim = U(:,1:n_pc) * S(1:n_pc,1:n_pc) / sqrt(double(m));

    simData.yStd = ysimStd;
    simData.x = x;
    simData.Ksim = Ksim;
    
    % Set up GPMSA
    paramout = setupModel([], simData);

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

        % Similar, but construct covariance with "addResidVar" option
        rng(seed,'twister')
        mcmc.model.debugRands=true;
        pred_arv=gPredict([0.5,0.5],mcmc.pvals(1:n_pred),mcmc.model,mcmc.data, ...
                        'addResidVar',1,'returnMuSigma',1);
        pred_arv_w=pred_arv.w;
        pred_arv_Myhat=pred_arv.Myhat;
        pred_arv_Syhat=[pred_arv.Syhat{:}];
    else
        pred_w = [];
        pred_Myhat = [];
        pred_Syhat = [];
        pred_arv_w = [];
        pred_arv_Myhat = [];
        pred_arv_Syhat = [];
    end

    % Store stuff needed for python in a struct
    res.y = y';
    res.ystd = ysimStd';
    res.xt = simData.x;
    res.y_ind = time';
    res.K = Ksim;
    res.ll = ll;
    res.ll_time = ll_time;
    res.mcmc = mcmc_out;
    res.mcmc_time = mcmc_time;
    res.pred_w = pred_w;
    res.pred_Myhat = pred_Myhat;
    res.pred_Syhat = pred_Syhat;
    res.pred_arv_w = pred_arv_w;
    res.pred_arv_Myhat = pred_arv_Myhat;
    res.pred_arv_Syhat = pred_arv_Syhat;
    
end
