load('multi_ex_sim.mat')

% simData object
simData.Ksim = K';
simData.yStd = y_std';
simData.x = [x_trans t_trans];

load('multi_ex_obs.mat')

% obsData object
obsData.Kobs = K';
obsData.yStd = y_std';
obsData.x = [0.5];
obsData.Dobs = D';

model = setupModel(obsData, simData);

rand('twister', 42.);

% model = gpmmcmc(model, 1000, 'step', 1); 
% 
% theta_draws = [model.pvals.theta]';
% 
% subplot(131)
% histogram(theta_draws(:, 1))
% xlim([0, 1])
% subplot(132)
% histogram(theta_draws(:, 2))
% xlim([0, 1])
% subplot(133)
% histogram(theta_draws(:, 3))
% xlim([0, 1])

model = stepsize(model, 50, 20);

model.mcmc.thetawidth
model.mcmc.rhoVwidth
model.mcmc.rhoUwidth
model.mcmc.lamVzwidth
model.mcmc.lamUzwidth
model.mcmc.lamWswidth
model.mcmc.lamWOswidth
model.mcmc.lamOswidth

model.mcmc.thetawidth = [0.32851417 0.19923462 0.29524287];
model.mcmc.rhoVwidth = [0.64229304];
model.mcmc.rhoUwidth = [0.73041073 0.3164296  0.4942496  0.22727609 0.52874024 ...
                        0.06390931 0.10111105 0.21261548 0.11987422 0.25976061 ...
                        0.14989501 0.17766716 0.17397863 0.4371333  0.21922118 ...
                        0.09707957 0.12826382 0.21741099 0.26190697 0.3824893 ];
model.mcmc.lamVzwidth = [0.89249099];
model.mcmc.lamUzwidth = [0.7276924  0.42184327 0.24459325 0.11805602 0.09055511];
model.mcmc.lamWswidth = [17284.82825878 15964.80901439 13384.19704841  8707.71618808 4512.28983889];
model.mcmc.lamWOswidth = [219.21593629];

model = gpmmcmc(model, 1000, 'step', 1); 

theta_draws = [model.pvals.theta]';

subplot(131)
histogram(theta_draws(1000:end, 1))
xlim([0, 1])
subplot(132)
histogram(theta_draws(1000:end, 2))
xlim([0, 1])
subplot(133)
histogram(theta_draws(1000:end, 3))
xlim([0, 1])

