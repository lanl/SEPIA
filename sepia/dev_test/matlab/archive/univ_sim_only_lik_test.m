% Test case for comparing sepia/GPMSA likelihood on univariate, sim-only

addpath(genpath('GPMSAmatlab'))
%% set up data (like sepiaDevTest1)

m = 300;
x = 0.5 * ones(m, 1);
t = linspace(0, 1, m)';
y = 10 .* t.^2 .* cos(10.*t);
y_std = y - mean(y);

y_sd = std(y_std);
y_std = y_std/y_sd;

%% Make structure for model
simData.x = [x, t]; 
simData.yStd = y_std';
simData.Ksim = [1];
data.simData = simData;

fprintf('ready to make gpmsa model\n')


%% Set up GPMSA
paramout = setupModel([], data.simData);

%% Call log lik
C.var='all';
%tic;
tmod = computeLogLik(paramout.model,paramout.data,C);
%ll_time = toc;
ll = tmod.logLik;

%% time 1000 times
tic;
for i=1:1000
    tmod = computeLogLik(paramout.model,paramout.data,C);
end
ll_time = toc;

%% Save to compare with python
save('-v7', 'data/univ_sim_only_lik_test.mat');