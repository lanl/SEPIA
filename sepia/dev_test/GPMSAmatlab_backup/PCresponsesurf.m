function pred = PCresponsesurf(pout, pvec)
% show how sensitive the simulator output is to changes in x and to changes
% in theta

% extract the stuff we need from pout
model = pout.model;
data = pout.data;
pvals = pout.pvals(pvec);

% p = model.p; % length of x
% q = model.q; % length of theta
% % (these are both 1 for this example)

% Set up the prediction grid
grid = 0:0.075:1;
[gridx, gridy] = meshgrid(grid, grid);
npc = size(pout.simData.Ksim, 2); % number of principal components

% AzEl = [45 55]; % azimuth and elevation, specifying the viewpoint of the 3-D plot we'll make below

% make the prediction at these grid points
xpred = gridx(:); 
theta = gridy(:);
mode='wpred';

if ~isempty(data.ztSep) mode='uvpred'; end
pred = gPred(xpred, pvals, model, data,mode, theta);
if ~isempty(data.ztSep) pred.w=pred.u; end
pm = squeeze(mean(pred.w, 1));



% make a surface plot for each principal component
figure
for ii = 1:npc
    w1 = reshape(pm(ii, :), [14 14]); % make the mean for component ii match the grid size
    
%     subplot(npc, 1, ii);
    subplot(1, npc, ii);
    mesh(gridx, gridy, w1); 
    axis square;
    hold on;
    xlabel('x','fontSize',11);
    ylabel('\theta','fontSize',11);
    zlabel(['w_' num2str(ii) '(x,\theta)'],'fontSize',11);
    title(['PC ' num2str(ii)]);
end
