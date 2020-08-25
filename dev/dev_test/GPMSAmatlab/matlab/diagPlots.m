% function ret=diagPlots(pout,pvec,plotNum,varargin)
% Some generic plots for GPS/SA diagnostics. Note that most plots of
% response surfaces and predictions are application specific because of
% the unknown structure of the data. (see basicExPlots for examples)
%   pout - the structure output from a gaspMCMC function call
%   pvec - a vector of the elements of the associated MCMC chain to process
%   plotnum - which plot (1-8) to do; scalar.
%      1 - rho plot for the theta and x variables by PC
%      101 - summary rho plot across all PC's (weighted by var contrib.)
%      2 - theta calibration plot
%      3 - lamOS and lamVz combined s.d. (joint model capable)
%      4 - PC diagnostics from the simulation dataset (for PC settings analysis)
%      5 - 1D conditional plot eta MAP mean and pointwise +/-2sd,
%          non-active vars at mid-range (for scalar response model)
%      6 - 2D conditional eta MAP mean response, non-active vars at mid-range,
%          default vars [1,2], specify with 'vars2D' optional arg
%          (for scalar response model)
%   TBD - Response plot of the simulation model basis loadings vs. params
%   Possible variable/value sets:
%    'labels' - cell array of labels for input variable names
%    'labels2' - cell array of labels for output variable names
%    'figNum' - figure number to plot into
%    'evenWeight' - do weighting calculations evenly (no PCs weighting)
%    'ngrid' - pass to gPlotMatrix
%    'ksd' - pass to gPlotMatrix
%    'standardized' - output variables in standardized scale
%    'var1D' - variables to be varied in a conditional plot
%              (type 5), default 1
%    'vars2D' - 2-vector of variables to be varied in a conditional plot
%              (type 6), default [1 2]
%    'gridCond' - prediction grid for conditional plot (type 5 or 6), 
%               default is linspace(0,1,10)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ret=diagPlots(pout,pvec,plotNum,varargin)
 ret=[];
 % Extract some variables from the structure, for convenience
  pvals=pout(1).pvals(pvec);
  model=pout(1).model; data=pout(1).data; 
  obsData=pout(1).obsData; simData=pout(1).simData;
  pu=model.pu; pv=model.pv; p=model.p; q=model.q;n=model.n;
  py=size(obsData(1).Dobs,1);
  % set defaults, then parse arguments
  labels=[];
  for ii=1:p; labels=[labels {['x ' num2str(ii)]}]; end
  for ii=1:q; labels=[labels {['theta ' num2str(ii)]}]; end
  closeFig=false; figNum=plotNum; 
  evenWeight=false;
  ngrid=40; ksd=0.05;
  standardized=true;
  for ii=1:py; labels2{ii}=['Var ',num2str(ii)]; end
  vars2D=[1 2];
  var1D=1;
  gridCond=linspace(0,1,10);
  parseAssignVarargs({'labels','figNum','closeFig','evenWeight', ...
                       'ngrid','ksd','standardized','labels2', ...
                       'var1D','vars2D','gridCond'});

% Plot the rhoU response in box plots, by principal component
% The rhoU indicates the degree to which the variable affects the
%  principal component. rho is in (0,1), rho=1 indicates no effect.
  if plotNum==1;
    figure(figNum); clf;
    % Collect the MCMC record of the betas
      bu=[pvals.betaU]';
    % Transform them into rhos
      ru = exp(-bu*0.25);
    % set up labels for the plot
    for ii=1:pu;
        r=ru(:,(ii-1)*(p+q)+1:ii*(p+q));
        subplot(pu,1,ii);
        gBoxPlot(r,'labels',labels);
        title(['PC' num2str(ii)]);
        a=axis; axis([a(1) a(2) 0 1]);
    end
  end

% boxplot the mean rhoU response across PCs
  if plotNum==101
    figure(figNum); clf;
    % recreate the PC variability
      [U,S,V]=svd(simData.yStd,0);
      a=diag(S).^2; a=a./sum(a); a=a(1:pu);
    % Collect the MCMC record of the betas
      bu=[pvals.betaU]';
    % get the weighted mean of each across PCs
      bumean=zeros(length(pvec),p+q);
      aweight=repmat(a,1,p+q)/sum(a);
      if evenWeight; aweight=ones(pu,p+q)/pu; end
      for ii=1:length(pvec)
        bumean(ii,:)=sum( reshape(bu(ii,:),p+q,pu)' .* aweight );
      end
    % plot the means transformed into rhos
      gBoxPlot(exp(-0.25*bumean),'labels',labels); 
      a=axis; axis([a(1:2) 0 1]);
      ret.bwm=bumean;
  end

% Examine the theta posterior calibration
% Each theta was estimated with MCMC, the result is a sample of the
%   underlying theta distribution
  if plotNum==2
    figure(figNum); clf;
    t=[pvals.theta]';
    gPlotMatrix(t,'shade',1,'lstyle','imcont','ngrid',ngrid, ...
                  'ksd',ksd,'shade',1,'labels',labels);
  end

% Plot the lamOS and lamVz combined s.d. These together indicate 
%    how much the simulation model is regularized, corresponds to how
%    important the simulation data is. This is particularly interesting
%    in joint models, or models with lamVz groups.
  if plotNum==3
    figure(figNum); clf;
    lovSD=[]; L={};
    for ii=1:length(pout)
       los=[pout(ii).pvals(pvec).lamOs]';
       lvz=[pout(ii).pvals(pvec).lamVz]';
       lovSD=[lovSD sqrt(1./repmat(los,1,size(lvz,2)) + 1./lvz) ];
       for jj=1:size(lvz,2)
         L{end+1}=['Mod ' num2str(ii) ' Grp ' num2str(jj)];
       end
    end
    boxplot(lovSD,'labels',L);
    a=axis; axis([a(1:2) 0 max(a(4),1)]);

  end

% Plot the calibrated discrepancy, each output point as a 
% discrete response in its own histogram
  if plotNum==4
    figure(figNum); clf
    % predict in uvpred mode over the specified realizations
    pred=gPred(0.5,pvals,pout.model,pout.data,'uvpred');
    v=(pred.v * obsData(1).Dobs)';
    if ~standardized
      if isscalar(simData.orig.ysd)
        v=v.* simData.orig.ysd;
      else
        v=v.*repmat(simData.orig.ysd,1,size(v,2));
      end
      v=v+repmat(simData.orig.ymean,1,size(v,2));
    end
    isize=ceil(sqrt(py)); jsize=ceil(py/isize);
    for ii=1:py
      ret.h(ii)=gPackSubplot(isize,jsize,ii,0,0.4); hold on;
      hist(v(ii,:));
      a=axis;
      text(a([1 2])*[0.9 0.1]',a([3 4])*[0.1 0.9]',labels2{ii});
    end
    
  end

% plot a 1D conditional response mean plot of the simulation emulator (eta).
  if plotNum==5
    figure(figNum);
    [mapVal pvecMAP]=max([pout.pvals(pvec).logPost]);
    if n>0
      xpred=0.5*ones(numel(gridCond),q);
      xpred(:,var1D)=gridCond;
      pred=gPredict(0.5*ones(numel(gridCond),1),...
             pout.pvals(pvec(pvecMAP)),pout.model,pout.data, ...
            'theta',xpred,'mode','wpred','returnMuSigma',1);
    else
      xpred=0.5*ones(numel(gridCond),p);
      xpred(:,var1D)=gridCond;
      pred=gPredict(xpred,pout.pvals(pvec(pvecMAP)),pout.model,pout.data, ...
            'returnMuSigma',1);
    end
    plot(gridCond,pred.Myhat); hold on;
    plot(gridCond,pred.Myhat+sqrt(diag(pred.Syhat{1}))'*2,':');
    plot(gridCond,pred.Myhat-sqrt(diag(pred.Syhat{1}))'*2,':');
    mean(pred.Myhat)
    xlabel(labels{var1D});
  end
  
% plot a 2D conditional response mean plot of the simulation emulator (eta).
  if plotNum==6
    figure(figNum); 
    [mapVal pvecMAP]=max([pout.pvals(pvec).logPost]);
    [g1 g2]=meshgrid(gridCond); 
    if n>0
      xpred=0.5*ones(numel(gridCond)^2,q);
      xpred(:,vars2D(1))=g1(:); xpred(:,vars2D(2))=g2(:);
      pred=gPredict(0.5*ones(numel(gridCond),1),...
           pout.pvals(pvec(pvecMAP)),pout.model,pout.data, ...
           'theta',xpred,'returnMuSigma',1);
    else
      xpred=0.5*ones(numel(gridCond)^2,p);
      xpred(:,vars2D(1))=g1(:); xpred(:,vars2D(2))=g2(:);
      pred=gPredict(xpred,pout.pvals(pvec(pvecMAP)),pout.model,pout.data, ...
          'returnMuSigma',1);
    end
    mesh(g1,g2,reshape(pred.Myhat,size(g1)));
    xlabel( labels{vars2D(1)} );
    ylabel( labels{vars2D(2)} );
  end

if closeFig; close(figNum); end
  
end %main plot function
