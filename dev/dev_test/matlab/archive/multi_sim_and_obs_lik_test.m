% Test case for comparing sepia/GPMSA likelihood on univariate, sim-only

addpath(genpath('GPMSAmatlab'))
%% set up data (using simData portion only)
data = gen_data_ex2(2);

% Get sim data for python
y_sim = data.simData.orig.y;
ymean = data.simData.orig.ymean;
ysd = data.simData.orig.ysd;
y_ind_sim = data.simData.orig.time;
ystd = data.simData.yStd;
x_sim = data.simData.x;
Ksim = data.simData.Ksim;

% Get obs data for python
x_obs = zeros(length(data.obsData), 1);
y_obs = zeros(length(data.obsData), 10);
y_ind_obs = data.obsData(1).orig.time;
for i=1:3
    y_obs(i, :) = data.obsData(i).orig.y;
    x_obs(i) = data.obsData(i).x;
end
Dsim = data.obsData(1).Dobs;

fprintf('ready to make gpmsa model\n')


%% Set up GPMSA
paramout = setupModel(data.obsData, data.simData);

% To match python behavior when init lamWOs out of bounds
if paramout.model.lamWOs > paramout.priors.lamWOs.bUpper
    paramout.model.lamWOs = paramout.priors.lamWOs.bUpper - 1;
end

%% Call log lik
C.var='all';
tic;
tmod = computeLogLik(paramout.model,paramout.data,C);
toc
ll = tmod.logLik;

%% time 1000 times
tic;
for i=1:1000
    tmod = computeLogLik(paramout.model,paramout.data,C);
end
ll_time = toc;

%% Save to compare with python
save('-v7', 'data/multi_sim_and_obs_lik_test.mat');