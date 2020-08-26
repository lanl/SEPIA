% Sets up test case for comparing multi sim and obs with hier theta to python

function res = setup_multi_sim_and_obs_hiertheta(m, n, nt_sim, nt_obs, noise_sd, nx, n_pc, seed, n_lik, n_mcmc, n_pred, n_shared)

    fprintf('\nStarting matlab setup_multi_sim_and_obs_sharedtheta.m\n')

    addpath(genpath('GPMSAmatlab'))

    rng(seed,'twister');
    
    for mi = 1:n_shared
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
        simData.orig.y = y;
        simData.orig.time = time;

        for i=1:n
          obsData(i).orig.y = y_obs;
          obsData(i).ymean=interp1(time',ymean,time_obs');
          obsData(i).yStd=(y_obs(:, i)-obsData(i).ymean)/ysimsd;
          obsData(i).Kobs=zeros(length(time_obs),n_pc);
          obsData(i).time = time_obs';
          for j=1:n_pc
            obsData(i).Kobs(:,j)=interp1(time',Ksim(:,j),obsData(i).time);
          end
          obsData(i).x = x_obs(i, 1);
          obsData(i).orig.all_x = x_obs(:, 1);
          obsData(i).Dobs = [];
          obsData(i).orig.time = time_obs;
        end

        % Set up GPMSA
        paramout(mi) = setupModel(obsData, simData);

    end

    % Setup hier params; a single hier theta shared across all models
    for mi = 1:n_shared
        hierParams(1).vars(mi).modNum=mi;
        hierParams(1).vars(mi).varNum=1;
    end
    % a starting point and a stored location for the hierarchical model
    % the hierarchical distribution is a normal, with mean and precision
    hierParams(1).model.mean=0.5;
    hierParams(1).model.lam=5;    % for grouping up

    % priors for the hierarchical parameters
    % the mean is from a normal dist, the lam is from a gamma
    hierParams(1).priors.mean.mean=0.5;
    hierParams(1).priors.mean.std=10;
    hierParams(1).priors.mean.bLower=0;
    hierParams(1).priors.mean.bUpper=1;
    hierParams(1).priors.lam.a=1;
    hierParams(1).priors.lam.b=1e-3;
    hierParams(1).priors.lam.bLower=0;
    hierParams(1).priors.lam.bUpper=Inf;

    % and a place for mcmc control parameters
     hierParams(1).mcmc.meanWidth=0.2;
     hierParams(1).mcmc.lamWidth=100;
     % lockstep update parameters
     hierParams(1).mcmc.lockstepMeanWidth=0.2;
     % lambda will be sampled as an adaptive parameter

    % a place for recording the samples, in the pvals structure
    hierParams(1).pvals.mean=[];
    hierParams(1).pvals.lam=[];
    
    %% Do likelihood calls
    ll = 0;
    ll_time = 0;
    if n_lik > 0
        fprintf('doing matlab likelihood calls\n')
        for mi = 1:n_shared
            C.var='all';
            ll_tmp = computeLogLik(paramout(mi).model,paramout(mi).data,C).logLik;
            tic;
            for i = 1:n_lik
                computeLogLik(paramout(mi).model,paramout(mi).data,C);
            end
            ll_time = toc;
            ll = ll + ll_tmp;
        end
    end
    
    %% Do mcmc samples
    rng(seed,'twister');
    if n_mcmc > 0
        tic;
        [mcmc, hier_mcmc] = gpmmcmc(paramout, n_mcmc, 'hierParams', hierParams);
        mcmc_time = toc;
        mcmc_out.hier_mean = [hier_mcmc.pvals.mean]';
        mcmc_out.hier_lam = [hier_mcmc.pvals.lam]';
        mcmc_out.betaU = [mcmc(1).pvals.betaU]';
        %mcmc_out.betaV = [mcmc(1).pvals.betaV]';
        mcmc_out.theta = [mcmc(1).pvals.theta]';
        mcmc_out.lamUz = [mcmc(1).pvals.lamUz]';
        %mcmc_out.lamVz = [mcmc(1).pvals.lamVz]';
        mcmc_out.lamWs = [mcmc(1).pvals.lamWs]';
        mcmc_out.lamWOs = [mcmc(1).pvals.lamWOs]';
        mcmc_out.lamOs = [mcmc(1).pvals.lamOs]';
        mcmc_out.logPost = [mcmc(1).pvals.logPost]';
        for mi = 2:n_shared
            mcmc_out.betaU = cat(3, mcmc_out.betaU, [mcmc(mi).pvals.betaU]');
            %mcmc_out.betaV = cat(3, mcmc_out.betaV, [mcmc(mi).pvals.betaV]');
            mcmc_out.theta = cat(3, mcmc_out.theta, [mcmc(mi).pvals.theta]');
            mcmc_out.lamUz = cat(3, mcmc_out.lamUz, [mcmc(mi).pvals.lamUz]');
            %mcmc_out.lamVz = cat(3, mcmc_out.lamVz, [mcmc(mi).pvals.lamVz]');
            mcmc_out.lamWs = cat(3, mcmc_out.lamWs, [mcmc(mi).pvals.lamWs]');
            mcmc_out.lamWOs = cat(3, mcmc_out.lamWOs, [mcmc(mi).pvals.lamWOs]');
            mcmc_out.lamOs = cat(3, mcmc_out.lamOs, [mcmc(mi).pvals.lamOs]');
            mcmc_out.logPost = cat(3, mcmc_out.logPost, [mcmc(mi).pvals.logPost]');
        end
    else
        mcmc_time = 0;
        mcmc_out = [];
    end

%     if n_pred > 0
%         %Predictions for comparison
%         p=mcmc;
%         rng(seed,'twister')
%         p.model.debugRands=true;
%         pred=gPredict([0.5],p.pvals(1:n_pred),p.model,p.data, ...
%                         'returnMuSigma',1,'mode','wpred');
%         pred_w=pred.w;
%         pred_Myhat=pred.Myhat;
%         pred_Syhat=[pred.Syhat{:}];
% 
%         %Predictions for comparison
%         rng(seed,'twister')
%         p.model.debugRands=true;
%         pred2=gPredict([0.5],p.pvals([1 5]),p.model,p.data, ...
%                         'returnMuSigma',1);
%         pred2_u=pred2.u;
%         pred2_v=pred2.v;
%         pred2_Myhat=pred2.Myhat;
%         pred2_Syhat=[pred2.Syhat{:}];
%     else
%          pred_w = [];
%          pred_Myhat = [];
%          pred_Syhat = [];
%          pred2_u = [];
%          pred2_v = [];
%          pred2_Myhat = [];
%          pred2_Syhat = [];
%     end
    
    % Store stuff needed for python in a struct
    res.y = paramout(1).simData.orig.y';
    res.ystd = paramout(1).simData.yStd';
    res.xt = paramout(1).simData.x;
    res.y_ind = paramout(1).simData.orig.time';
    res.K = paramout(1).simData.Ksim;
    res.y_obs = paramout(1).obsData(1).orig.y';
    res.x_obs = paramout(1).obsData(1).orig.all_x;
    res.y_ind_obs = paramout(1).obsData(1).orig.time';
    for mi = 2:n_shared
        res.y = cat(3, res.y, paramout(mi).simData.orig.y');
        res.ystd = cat(3, res.ystd, paramout(mi).simData.yStd');
        res.xt = cat(3, res.xt, paramout(mi).simData.x);
        res.y_ind = cat(3, res.y_ind, paramout(mi).simData.orig.time');
        res.K = cat(3, res.K, paramout(mi).simData.Ksim);
        res.y_obs = cat(3, res.y_obs, paramout(mi).obsData(1).orig.y');
        res.x_obs = cat(3, res.x_obs, paramout(mi).obsData(1).orig.all_x);
        res.y_ind_obs = cat(3, res.y_ind_obs, paramout(mi).obsData(1).orig.time');
    end
    res.ll = ll;
    res.ll_time = ll_time;
    res.mcmc = mcmc_out;
    res.mcmc_time = mcmc_time;
%     res.pred_w = pred_w;
%     res.pred_Myhat = pred_Myhat;
%     res.pred_Syhat = pred_Syhat;
%     res.pred2_u = pred2_u;
%     res.pred2_v = pred2_v;
%     res.pred2_Myhat = pred2_Myhat;
%     res.pred2_Syhat = pred2_Syhat;

    
end
