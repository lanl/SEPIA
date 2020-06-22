clear; clc;
addpath(genpath('test/GPMSAmatlab'))
addpath(genpath('data'))

% Read the data and setup the model;
[oD, sD] = readFeatures;

plot(sD.Ksim)
plot(sD.yStd)
plot(sD.orig.ymean)
plot(sD.orig.y)
sD.orig.ysd

nmcmc = 10;

% Save data for Python
y_sim = sD.orig.y;
xt_sim = sD.x;
pu = size(sD.Ksim, 2);
x_obs = oD.x;
Sigy = oD.Sigy;
y_obs = oD.orig.y;
Dobs = oD.Dobs;

lamWs_upper = Inf;
lamWs_a = 1;
lamWs_b = 0;
save('-v7', 'data_testcase/data/alal_data.mat', 'y_sim', 'xt_sim', 'pu', 'x_obs', 'Sigy', 'y_obs', 'Dobs', 'nmcmc', 'lamWs_upper', 'lamWs_a', 'lamWs_b');

alCal = setupModel(oD, sD);
alCal.priors.lamWs.params = [ones(alCal.model.pu,1), zeros(alCal.model.pu,1)];
alCal.priors.lamWs.bUpper = Inf;

% Compute good step sizes for the MCMC -- for comparing to sepia, don't do this
%nburn = 100; nlev = 25;
%alCal = stepsize(alCal, nburn, nlev);

rand('twister', 42.);

% MCMC
tic;
%alCal = gpmmcmc(alCal, nmcmc, 'step', 1); 
alCal = gpmmcmc(alCal, nmcmc); 
mcmc_time = toc;
save('data_testcase/data/alCal.mat', 'alCal');
%load('alCal.mat', 'alCal');


betaU_samp = [alCal.pvals.betaU]';
betaV_samp = [alCal.pvals.betaV]';
lamUz_samp = [alCal.pvals.lamUz]';
lamVz_samp = [alCal.pvals.lamVz]';
lamWs_samp = [alCal.pvals.lamWs]';
lamWOs_samp = [alCal.pvals.lamWOs]';
lamOs_samp = [alCal.pvals.lamOs]';
theta_samp = [alCal.pvals.theta]';
logPost_trace = [alCal.pvals.logPost]';
save('-v7', 'data_testcase/data/alal_result.mat', 'betaU_samp', 'betaV_samp', 'lamUz_samp', 'lamVz_samp', ...
            'lamWs_samp', 'lamWOs_samp', 'lamOs_samp', 'theta_samp', 'logPost_trace', 'mcmc_time');


% Plot stuff
theta = [alCal.pvals.theta]';
p = alCal.model.p;
q = alCal.model.q;
pq = p + q;
m = alCal.model.m;
pu = alCal.model.pu;
xmin = alCal.simData.orig.xmin;
xrange = alCal.simData.orig.xrange;
%xnames = alCal.simData.orig.xnames;
xnames = {'A', 'B', 'C', 'n', 'm', 'V_1', 'V_2', 'V_3', 'G_1', '\Delta_2', '\Delta_3'};
Ksim = alCal.simData.Ksim;
ymean = alCal.simData.orig.ymean;
ysd = alCal.simData.orig.ysd;
yobs = alCal.obsData.orig.y;
ysim = alCal.simData.orig.y;
%pvec = floor(linspace(2501, 12500, 100));
pvec = 1:100;

% Check the correlation coefficients.
rhoU = exp(-[alCal.pvals(pvec).betaU]/4)';
figure('Units', 'inches', 'Position', [0, 1, 11, 8.5]);
for i=1:pu
    subplot(ceil(pu/2),2,i);
    ix = (i-1)*pq + (1:pq);
    boxplot(rhoU(:,ix));
    ylim([0,1]);
end
saveas(gcf, 'data_testcase/rho', 'png');
close;

% Chains
figure;
for i=1:q
    subplot(q,1,i);
    plot(theta(:,i));
end
saveas(gcf, 'data_testcase/figures/thetaChain', 'png');
close;

[S,Ax,BigAx,H,Hax] = plotmatrix(alCal.simData.x(:,2:end), 'b.');
for i=1:length(xnames)
    Ax(1,i).Title.String = xnames{i};
    Ax(i,1).YLabel.String = xnames{i};
    Ax(i,1).YLabel.FontWeight = 'bold';
    Ax(i,1).YLabel.Color = [0,0,0];
    Ax(i,1).YTickLabel = {};
    Ax(end,i).XTickLabel = {};
end
saveas(gcf, 'data_testcase/figures/design', 'png');
close;

%figure('Units','inches','Position',[1,1,6,6]);
[S,Ax,BigAx,H,Hax] = plotmatrix(theta(2501:end,:));
for i=1:length(xnames)
    Ax(1,i).Title.String = xnames{i};
    Ax(i,1).YLabel.String = xnames{i};
    Ax(i,1).YLabel.FontWeight = 'bold';
    Ax(i,1).YLabel.Color = [0,0,0];
    Ax(i,1).YTickLabel = {};
    Ax(end,i).XTickLabel = {};
end
saveas(gcf, 'data_testcase/figures/thetaCal', 'png');
%saveas(gcf, 'thetaCalBig', 'png');
close;

% % Cross-validation
% cvpred = zeros(length(ymean),m);
% for i=1:m
%     disp(i);
%     ixCV = setdiff(1:m, i);
%     simDataCV = alCal.simData;
%     simDataCV.x = simDataCV.x(ixCV,:);
%     simDataCV.yStd = simDataCV.yStd(:,ixCV);
%     emuCV = setupModel([], simDataCV);
%     yCV = gPredict(alCal.simData.x(i,:), alCal.pvals(pvec), emuCV.model, emuCV.data);
%     yCV = ysd * alCal.simData.Ksim * yCV.w';
%     yCV = bsxfun(@plus, yCV, ymean);
%     cvpred(:,i) = mean(yCV,2);
% end
% save('cvpred.mat', 'cvpred');
% %load('cvpred.mat', 'cvpred');
% 
% for i=1:length(ymean)
%     figure;
%     plot(ysim(i,:), cvpred(i,:), '.b');
%     title(['Feature ', num2str(i)], 'FontSize', 14);
%     xlabel('Sim', 'FontSize', 14);
%     ylabel('CV Prediction', 'FontSize', 14);
%     axis manual
%     line
%     saveas(gcf, ['cv',num2str(i)], 'png');
%     close;
% end