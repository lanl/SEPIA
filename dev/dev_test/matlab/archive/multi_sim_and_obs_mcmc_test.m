% Test case for comparing sepia/GPMSA mcmc on multivariate, sim and obs
function multi_sim_and_obs_mcmc_test(nsamp, nburn, seed, lamWOs_init, n_pc, do_stepsize)

    addpath(genpath('GPMSAmatlab'))

    %% set up data (using simData portion only)
    data = gen_data_ex2(n_pc);
    
    rand('twister', seed);

    % Get sim data for python
    y = data.simData.orig.y;
    ymean = data.simData.orig.ymean;
    ysd = data.simData.orig.ysd;
    y_ind = data.simData.orig.time;
    ystd = data.simData.yStd;
    x = data.simData.x;
    Ksim = data.simData.Ksim;
    
    % Get obs data for python
    y_obs = zeros(3, 10);
    y_obs_mean = zeros(3, 10);
    y_obs_std = zeros(3, 10);
    x_obs = zeros(3, 1);
    for i=1:3
        y_obs(i, :) = data.obsData(i).orig.y;
        y_obs_mean(i, :) = data.obsData(i).orig.ymean;
        x_obs(i) = data.obsData(i).x;
        y_obs_std(i, :) = data.obsData(i).yStd;
    end
    % All same time index for this case
    Kobs = data.obsData(1).Kobs;
    Dobs = data.obsData(1).Dobs;
    y_ind_obs = data.obsData(1).orig.time;

    fprintf('ready to make gpmsa model\n')

    %% Set up GPMSA
    %for listToSample=1:4

    paramout = setupModel(data.obsData, data.simData);

    if lamWOs_init > 0
        paramout.model.lamWOs = lamWOs_init;
    end
    %paramout.priors.lamWOs.bUpper = 500000;


    % quick check: save off the initial struct, logLik, logprior
    paramout=gpmmcmc(paramout,0,'initOnly',1);
    initial_LL=paramout.model.logLik;
    initial_LPR=computeLogPrior(paramout.priors,paramout.mcmc,paramout.model);
    initial_LPR=initial_LPR.logPrior;
    
    % For testing step size
    if do_stepsize == 1
        step_burn = 100;
        step_levels = 8;
        paramout = stepsize(paramout, step_burn, step_levels);
        %fprintf('step sizes:')
        %paramout.mcmc
        tic;
        paramout = gpmmcmc(paramout, nburn+nsamp, 'step', 1);
        mcmc_time = toc;
    else
        tic;
        paramout = gpmmcmc(paramout, nburn+nsamp);
        mcmc_time = toc;
    end
    
    tic;
    paramout = gpmmcmc(paramout, nburn+nsamp);
    mcmc_time = toc;

    %figure(listToSample); showPvals(paramout.pvals)
    %end
    %% Extract results
    betaU_samp = [paramout.pvals.betaU]';
    betaV_samp = [paramout.pvals.betaV]';
    lamUz_samp = [paramout.pvals.lamUz]';
    lamVz_samp = [paramout.pvals.lamVz]';
    lamWs_samp = [paramout.pvals.lamWs]';
    lamWOs_samp = [paramout.pvals.lamWOs]';
    lamOs_samp = [paramout.pvals.lamOs]';
    theta_samp = [paramout.pvals.theta]';
    logPost_trace= [paramout.pvals.logPost]';

    %Predictions for comparison
    p=paramout;
    rng(seed,'twister')
    p.model.debugRands=true;
    pred=gPredict([0.5],p.pvals(1:5),p.model,p.data, ...
                    'returnMuSigma',1,'mode','wpred');
    pred_w=pred.w;
    pred_Myhat=pred.Myhat;
    pred_Syhat=[pred.Syhat{:}];

    %Predictions for comparison
    rng(seed,'twister')
    p.model.debugRands=true;
    pred2=gPredict([0.5],p.pvals([1 5]),p.model,p.data, ...
                    'returnMuSigma',1);
    pred2_u=pred2.u;
    pred2_v=pred2.v;
    pred2_Myhat=pred2.Myhat;
    pred2_Syhat=[pred2.Syhat{:}];

    %% Save to compare with python
    save('-v7', 'data/multi_sim_and_obs_mcmc_test.mat');

end