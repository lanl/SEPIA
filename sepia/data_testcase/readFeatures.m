function [obsData, simData] = readFeatures
%clear; clc;

datadir = 'data_testcase/data/';

% Read in the design and standardize it.
design = importdata([datadir, 'Al.trial5.design.txt']);
xnames = design.colheaders;
x = design.data;
xmin = min(x);
xmax = max(x);
xrange = xmax - xmin;
x = bsxfun(@minus, x, xmin);
x = bsxfun(@rdivide, x, xrange);

m = size(x,1);

% Append a fake experimental variable
x = [0.5*ones(m,1), x];
xmin = [0, xmin];
xrange = [1, xrange];

% Read in the data
shot = {'104S', '105S', '106S'};
fcount = 4;
ysim = zeros(fcount*length(shot), size(x,1));
yobs = zeros(fcount*length(shot), 1);
for i=1:length(shot)
    raw = importdata([datadir, 'features_cdf', shot{i}, '.csv']);
    rawobs = importdata([datadir, 'features_cdf_obs', shot{i}, '.csv']);
    idx = (i-1)*fcount + (1:fcount); 
    % The 10000 puts the sims on the same scale as the data.
    ysim(idx,:) = 10000*raw.data(:,[4,6,8,10])';
    yobs(idx) = rawobs.data(:,[4,6,8,10])';
end

% Standardize the simulations
ysimmean = mean(ysim, 2);
ysimStd = bsxfun(@minus, ysim, ysimmean);
ysimsd = std(ysimStd(:));
ysimStd = ysimStd / ysimsd;
[U, S, ~] = svd(ysimStd, 0);
pu = 11;
Ksim = U(:,1:pu) * S(1:pu,1:pu) / sqrt(m);

% simData object
simData.Ksim = Ksim;
simData.yStd = ysimStd;
simData.x = x;
simData.orig.y = ysim;
simData.orig.ymean = ysimmean;
simData.orig.ysd = ysimsd;
simData.orig.xmin = xmin;
simData.orig.xmax = xmax;
simData.orig.xrange = xrange;
simData.orig.xnames = xnames;

% obsData
obsData.x = 0.5;
obsData.yStd = (yobs - ysimmean) / ysimsd;
obsData.Kobs = Ksim;
properror = repmat([.01; .01; .01; .01], length(shot), 1);
obsData.Sigy = diag((properror.*yobs).^2) / (ysimsd^2);
obsData.Dobs = zeros(length(ysimmean),1);
obsData.orig.y = yobs;

[h, ax, bixax] = gplotmatrix([ysim'; yobs'], [], [zeros(m,1); 1], 'br', '.o', [1,5], false);
for i=1:length(ysimmean)
    ax(13,i).YLim = [0,1.2 * max(h(i,i,1).Values)];
    h(i,i,2).DisplayStyle = 'bar';
    h(i,i,2).FaceColor = [1,0,0];
    h(i,i,2).EdgeColor = [1,0,0];
    for j=setdiff(1:length(ysimmean),i)
        h(i,j,2).MarkerFaceColor = [1,0,0];
        h(j,i,2).MarkerFaceColor = [1,0,0];
    end
end
saveas(gcf, 'data_testcase/figures/featuresSimsData.png');