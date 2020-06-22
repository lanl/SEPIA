% Sets up test case for comparing multi sim and obs to python

function res = setup_multi_sim_and_obs_noD(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc)

    fprintf('\nStarting matlab setup_multi_sim_and_obs_noD.m\n')

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
    
    y = zeros(m, nt_sim);
    time = linspace(0, 1, nt_sim);
    for i = 1:n_pc
        for j = 1:m
            y(j, :) = y(j, :) + wt_gen(j, i) * cos(pi * double(i) * time);
        end
    end
    y = y';
    
    % Generate obs data same way but with noise
    x_obs = repmat(linspace(0, 1, n), nx, 1)';
    beta = abs(0.5 * rand(nx, 1)) + 0.05;
    cov = zeros(n, n);
    for i = 1:n
        for j = 1:n
            cov(i, j) = exp(-sum(beta' .* (x_obs(i, :) - x_obs(j, :)).^2, 2));
        end
    end
    
    chCov = chol(cov + 1e-3 * eye(n));
    wt_gen = chCov' * randn(n, n_pc) .* 1./sqrt(double(cumsum(1:n_pc)));
    
    y_obs = zeros(n, nt_obs);
    time_obs = linspace(0, 1, nt_obs);
    for i = 1:n_pc
        for j = 1:n
            y_obs(j, :) = y_obs(j, :) + wt_gen(j, i) * cos(pi * double(i) * time_obs) + noise_sd * randn(1, nt_obs);
        end
    end
    y_obs = y_obs';
    
    
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
    
    for i=1:n
      obsData(i).ymean=interp1(time',ymean,time_obs');
      obsData(i).yStd=(y_obs(:, i)-obsData(i).ymean)/ysimsd;
      obsData(i).Kobs=zeros(length(time_obs),n_pc);
      obsData(i).time = time_obs';
      for j=1:n_pc
        obsData(i).Kobs(:,j)=interp1(time',Ksim(:,j),obsData(i).time);
      end
      obsData(i).x = x_obs(i, 1);
      obsData(i).Dobs = [];
    end
    
    % Set up GPMSA
    paramout = setupModel(obsData, simData);
    
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
        mcmc_out.theta = [mcmc.pvals.theta]';
        mcmc_out.lamUz = [mcmc.pvals.lamUz]';
        mcmc_out.lamWs = [mcmc.pvals.lamWs]';
        mcmc_out.lamWOs = [mcmc.pvals.lamWOs]';
        mcmc_out.lamOs = [mcmc.pvals.lamOs]';
        mcmc_out.logPost = [mcmc.pvals.logPost]';
    else
        mcmc_time = 0;
        mcmc_out = [];
    end
    
    % Store stuff needed for python in a struct
    res.y = y';
    res.ystd = ysimStd';
    res.xt = simData.x;
    res.y_ind = time';
    res.K = Ksim;
    res.y_obs = y_obs';
    res.x_obs = x_obs(:, 1);
    res.y_ind_obs = time_obs';
    res.K = Ksim;
    res.ll = ll;
    res.ll_time = ll_time;
    res.mcmc = mcmc_out;
    res.mcmc_time = mcmc_time;
    
    
end
