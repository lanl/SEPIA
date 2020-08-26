function h=gXval(pout,pvec,varargin)
% function handles=gXval(pout,pvec,varargin)
% leave-one-out cross-validation of eta emulator, using the pvals given in
%    the pout structure (i.e., no re-estimation of params)
%
% mode is a selection of:
%   'predict' - return holdout predictions as a gPredict-type struct array
%       fields are 'pred' predictions, 'predR predictions with resid.
%       error, and 'wdat', whe w's used in the model
%   'varExplained' - uses the samples to compute a variability explained by
%                    the PC reconstruction and the prediction
%   'PCplot' - (default) plot of each PC response.
%   'PCplotOrder' - plot of each PC response, in canonical order boxplot
%   'residErr' - prediction accuracy of multivariate response
%              (don't try this if it's highly multivariate)
%   'residSummary' - residual summary from each HO
%         (integrating over all multivariate responses, and all pvals)
% numSamp = number of samples to draw on xVal
% sampleGroups = cell array of groups to be held out in turn, 
%                rather than single-element holdout. Results are still
%                presented as individual holdout results rather than
%                than groupwise.
% % nreal = number of realizations to draw of each point (default 1)
% figNum = figure number to use (default varies by plot)
% standardized = in native space, whether results are standardized or
%                on the original scale (default is 1 = standardized scale)
% labels - sometimes works

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

simData=pout.simData;

% set up defaults
numSamp=size(simData.yStd,2);
mode='PCplot'; nreal=1; figNum=false; standardized=1; labels=[];
inCurrentAxes=false;
sampleGroups={};
verbose=1;
h=[];
parseAssignVarargs({'mode','figNum','numSamp','standardized','labels', ...
                    'inCurrentAxes','sampleGroups','verbose'});  

pvals=pout.pvals(pvec);
npv=length(pvec);
pu=size(simData.Ksim,2);

p=size(simData.Ksim,1);

m=size(simData.yStd,2);
if isempty(sampleGroups)
  m=size(simData.yStd,2);
  temp=ilinspace(1,m,numSamp);
  for ii=1:length(temp);
    msamp{ii}=temp(ii);
  end
else
  msamp=sampleGroups;
end
numHOGroups=length(msamp);

% do predictions for all display modes
simLoo=simData;
if verbose; fprintf('Predicting. '); counter('stime',1,numHOGroups,6,10); end
numSamp=0;
for ii=1:numHOGroups
  if verbose; counter(ii); end;
  simLoo.yStd=simData.yStd(:,setxor(1:m,msamp{ii}));
  simLoo.x=simData.x(setxor(1:m,msamp{ii}),:);
  pLoo=setupModel([],simLoo,[],'verbose',0);
  for jj=msamp{ii}(:)'
    numSamp=numSamp+1;
    pred(numSamp)=gPredict(simData.x(jj,:),pvals,pLoo.model,pLoo.data, ...
                  'returnMuSigma',1,'addResidVar',0);
    if strcmp(mode,'residErr') || strcmp(mode,'predict')
      predR(numSamp)=gPredict(simData.x(jj,:),pvals,pLoo.model,pLoo.data, ...
                  'returnMuSigma',1,'addResidVar',1);
    end
    wdat(numSamp,:)=pout.data.w(jj,:);
  end
end
if verbose; counter('end'); end

% choose plot type
switch(mode)
  
case 'predict'
  h.pred=pred;
  h.predR=predR;
  h.wdat=wdat;
  
case 'varExplained'
  % this has to be unscaled. 
    if isscalar(simData.orig.ysd)
      ysd=repmat(simData.orig.ysd,1,size(simData.orig.y,2));
    else
      ysd=simData.orig.ysd;
    end
    ymean=simData.orig.ymean;
  % create the three reconstructed dataset options
    ydat=simData.orig.y(msamp,:);
    yPC=ladd(lmult(wdat*simData.Ksim',ysd),ymean);
    % get the mean prediction
      mpredmat=zeros([ length(pred) size(pred(1).Myhat) ]);
      for ii=1:length(pred)
        mpredmat(ii,:,:)=pred(ii).Myhat;
      end
      mpred=squeeze(mean(mpredmat,2));
    yhat=ladd(lmult(mpred*simData.Ksim',ysd),ymean);
    fprintf('PC reconstruction explains %f of total data variability\n', ...
            sum(var(yPC))/sum(var(ydat)));
    fprintf('prediction mean explains %f of PC recon. variability\n', ...
            1-sum(var(yPC-yhat))/sum(var(yPC)) );
    fprintf('prediction mean explains %f of total data variability\n', ...
            1-sum(var(ydat-yhat))/sum(var(ydat)) );
    
case 'PCplot'
  if ~inCurrentAxes; if figNum; figure(figNum); else figure(61); end; clf; end
  isize=ceil(sqrt(pu)); jsize=ceil(pu/isize);
  pcpred=zeros(npv,numSamp);
  for pc=1:pu
    for ii=1:numSamp
      pcpred(:,ii)=mean(squeeze(pred(ii).w(:,pc)),2);
    end
    if ~inCurrentAxes
      h(pc)=gPackSubplot(isize,jsize,pc,0,0.1); 
    end
    hold on;
    plot(wdat(:,pc),mean(pcpred,1),'.');
    for ii=1:numSamp
      plot(wdat(ii,pc)*[1 1],gQuantile(pcpred(:,ii),[0.1 0.9]),...
           'linewidth',1.5);
      %plot(wdat(ii,pc)*[1 1],gQuantile(pcpred(:,ii),[0 1]),...
      %     'linewidth',1);
    end
    a=axis; a=min(a(1),a(3))*[1 0 1 0]+max(a(2),a(4))*[0 1 0 1]; axis(a);
    line(a([1 2]),a([3 4]));
    %text(a([1 2])*[0.9 0.1]',a([3 4])*[0.1 0.9]',['PC' num2str(pc)]);
    drawnow;
  end
  
case 'PCplotOrder'
  if ~inCurrentAxes; if figNum; figure(figNum); else figure(62); end; clf; end
  pcpred=zeros(npv,m);
  for pc=1:pu
    for ii=1:numSamp
      pcpred(:,ii)=mean(squeeze(pred(ii).w(:,pc)),2);
    end
    gPackSubplot(pu,1,pc,1,0);
    gBoxPlot(pcpred);
    drawnow
  end
  
case 'residErr'
  if ~inCurrentAxes; if figNum; figure(figNum); else figure(63); end; clf; end
  if isscalar(simData.orig.ysd)
    ysd=repmat(simData.orig.ysd',size(simData.orig.y,2),2);
  else
    ysd=repmat(simData.orig.ysd',1,2);
  end
  ymean=repmat(simData.orig.ymean',1,2);
  if standardized; yorig=simData.yStd;
              else yorig=simData.orig.y;
  end
  mipred=zeros(numSamp,p,2);
  mripred=zeros(numSamp,p,2);
  for ii=1:numSamp
    mv=zeros(npv,pu);
    for jj=1:length(pred(ii).Syhat); 
        mv(jj,:)=diag(pred(ii).Syhat{jj}); 
        mvR(jj,:)=diag(predR(ii).Syhat{jj}); 
    end
    micdf=(gGMICDF(pred(ii).Myhat',mv',[0.1 0.9])' * simData.Ksim')';
    mricdf=(gGMICDF(predR(ii).Myhat',mvR',[0.1 0.9])' * simData.Ksim')';
    if ~standardized
      micdf=(micdf.*ysd)+ymean;
      mricdf=(mricdf.*ysd)+ymean;
    end
    mipred(ii,:,:)=micdf;
    mripred(ii,:,:)=mricdf;
  end
  % now plot by response variable
  isize=ceil(sqrt(p)); jsize=ceil(p/isize);
  for ii=1:p
    if ~inCurrentAxes; h(ii)=gPackSubplot(isize,jsize,ii,0,0.1); end; hold on;
    %fprintf('Plot %d\n',ii);
    for jj=1:numSamp
      mi=squeeze(mipred(jj,ii,:));
      mri=squeeze(mripred(jj,ii,:));
      %fprintf('  '); fprintf('%10.6f ',mi);
      %fprintf(' | '); fprintf('%10.6f ',mri); fprintf('\n')
      plot(yorig(ii,msamp{jj}),mean(pred(jj).Myhat)*simData.Ksim(ii,:)','k+');
      plot(yorig(ii,msamp{jj})*[1 1],mri,'r','linewidth',1);
      plot(yorig(ii,msamp{jj})*[1 1],mi,'linewidth',1.5);
    end
    axis tight;
    a=axis; a=floor(min(a(1),a(3)))*[1 0 1 0]+ceil(max(a(2),a(4)))*[0 1 0 1]; 
    axis(a);
    line(a([1 2]),a([3 4]));
    if ~isempty(labels); label=labels{ii}; 
    else label=['var ' num2str(ii)];
    end
    text(a([1 2])*[0.9 0.1]',a([3 4])*[0.1 0.9]',label);
    drawnow;
  end
  

case 'residSummary'  
  if ~inCurrentAxes; if figNum; figure(figNum); else figure(64); end; clf; end
  if isscalar(simData.orig.ysd)
    ysd=simData.orig.ysd;
  else
    ysd=repmat(simData.orig.ysd,1,npv);
  end
  ymean=repmat(simData.orig.ymean,1,npv);
  if standardized; yorig=simData.yStd;
              else yorig=simData.orig.y;
  end
  for ii=1:numSamp
    y=(pred(ii).w*simData.Ksim')';
    if ~standardized
      y=(y.*ysd)+ymean;
    end
    tres=(y-repmat(yorig(:,msamp(ii)),1,npv));
    yres(:,ii)=tres(:);
  end
  gBoxPlot(yres,'noOutliers',1);
  tstr='residuals'; if standardized; tstr=['standardized ' tstr]; end
  tstr=[tstr ', MRSE=' num2str(mean(yres(:).^2))];
  title(tstr);
end



