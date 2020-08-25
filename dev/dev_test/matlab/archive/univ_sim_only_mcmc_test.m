% Test case for comparing sepia/GPMSA mcmc on univariate, sim-only

addpath(genpath('GPMSAmatlab'))
%% set up data (like sepiaDevTest1)

rand('twister', 42.);

m = 300;
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
simData.Ksim = [1];
data.simData = simData;

fprintf('ready to make gpmsa model\n')


%% Set up GPMSA
paramout = setupModel([], data.simData);

%% Do mcmc with default parameters
nsamp = 1000;
nburn = 100;
tic;
paramout = gpmmcmc(paramout, nburn+nsamp);
mcmc_time = toc;
betaU_samp = [paramout.pvals.betaU]';
lamUz_samp = [paramout.pvals.lamUz]';
lamWs_samp = [paramout.pvals.lamWs]';
lamWOs_samp = [paramout.pvals.lamWOs]';
logPost_trace= [paramout.pvals.logPost]';

%% make some predictions to test
  % basic samples prediction
    rng(42,'twister');
    paramout.model.debugRands=true;
    pred=gPredict([0.5,0.5],paramout.pvals(1:5),paramout.model,paramout.data,'returnMuSigma',1);
    pred_w=pred.w;
    pred_Myhat=pred.Myhat;
    pred_Syhat=[pred.Syhat{:}];
  % suitable for plotting
    rng(42,'twister')
    paramout.model.debugRands=true;
    nq=10; pred_plot_xpred= [0.5*ones(10,1) linspace(0,1,10)'];
    pred_plot=gPredict(pred_plot_xpred,paramout.pvals(100:100:1000),paramout.model,paramout.data);
    pred_plot_w=squeeze(pred_plot.w);

%% Save to compare with python
save('-v7', 'data/univ_sim_only_mcmc_test.mat');