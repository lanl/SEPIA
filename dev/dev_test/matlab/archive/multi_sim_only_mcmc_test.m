% Test case for comparing sepia/GPMSA mcmc on multivariate, sim-only
function multi_sim_only_mcmc_test(nsamp, nburn, listToSample, seed, lamWOs_init, n_pc)

    addpath(genpath('GPMSAmatlab'))

    %% set up data
    data = gen_data_ex(n_pc);
    
    rng(seed,'twister');

    % Get sim data for python
    y = data.simData.orig.y;
    ymean = data.simData.orig.ymean;
    ysd = data.simData.orig.ysd;
    y_ind = data.simData.orig.time;
    ystd = data.simData.yStd;
    x = data.simData.x;
    Ksim = data.simData.Ksim;

    fprintf('ready to make gpmsa model\n')

    %% Set up GPMSA
    %for listToSample=1:4

    paramout = setupModel([], data.simData);

    %%% Setting up to sample a subset of the variables
    %%% listToSample can be any set of  1,2,3,4
    %listToSample=1;  % Sample betaU only
    %listToSample=2;   % Sample lamUz only
    %listToSample=3;  % Sample lamWs only
    %listToSample=4;  % Sample lamWOs only
    listToSample = cell2mat(listToSample);
    % make all the lists consistent with the spec
    paramout.mcmc.pvars = paramout.mcmc.pvars([listToSample 5 6 7]);
    paramout.mcmc.svars = paramout.mcmc.svars(listToSample);
    paramout.mcmc.svarSize=paramout.mcmc.svarSize(listToSample);
    paramout.mcmc.wvars = paramout.mcmc.wvars(listToSample);

    if lamWOs_init > 0
        paramout.model.lamWOs = lamWOs_init;
    end
    %paramout.priors.lamWOs.bUpper = 500000;

    tic;
    paramout = gpmmcmc(paramout, nburn+nsamp);
    mcmc_time = toc;

    %figure(listToSample); showPvals(paramout.pvals)
    %end
    %% Extract results
    if isfield(paramout.pvals, 'betaU')
        betaU_samp = [paramout.pvals.betaU]';
    else
        betaU_samp = [];
    end
    if isfield(paramout.pvals, 'lamUz')
        lamUz_samp = [paramout.pvals.lamUz]';
    else
        lamUz_samp = [];
    end
    if isfield(paramout.pvals, 'lamWs')
        lamWs_samp = [paramout.pvals.lamWs]';
    else
        lamWs_samp = [];
    end
    if isfield(paramout.pvals, 'lamWOs')
        lamWOs_samp = [paramout.pvals.lamWOs]';
    else
        lamWOs_samp = [];
    end
    logPost_trace= [paramout.pvals.logPost]';

%% make some predictions to test
  % basic samples prediction
    rng(seed,'twister');
    paramout.model.debugRands=true;
    pred=gPredict([0.5,0.5],paramout.pvals(1:5),paramout.model,paramout.data,'returnMuSigma',1);
    pred_w=pred.w;
    pred_Myhat=pred.Myhat;
    pred_Syhat=[pred.Syhat{:}];

    % Similar, but construct covariance with "addResidVar" option
    paramout.model.debugRands=true;
    pred_arv=gPredict([0.5,0.5],paramout.pvals(1:5),paramout.model,paramout.data, ...
                    'addResidVar',1,'returnMuSigma',1);
    pred_arv_w=pred_arv.w;
    pred_arv_Myhat=pred_arv.Myhat;
    pred_arv_Syhat=[pred_arv.Syhat{:}];

    %% Save to compare with python
    save('-v7', 'data/multi_sim_only_mcmc_test.mat');

end