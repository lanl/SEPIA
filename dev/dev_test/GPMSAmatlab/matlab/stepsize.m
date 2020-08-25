% function [params hierParams] = stepsize(params,nBurn,nLev,varargin)
% compute step sizes from step size data collect run in gpmmcmc
% please see associated documentation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
%         Brian Williams, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [params hierParams] = stepsize(params,nBurn,nLev,varargin)

numMods=length(params);
clist=zeros(0,numMods);
hierParams=[];
parseAssignVarargs({'clist','hierParams'});
numHMods=length(hierParams);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up ranges, and precompute structures for quick&easy sampling

fprintf('Setting up structures for stepsize statistics collect ...\n');

ex = -(nLev-1)/2:(nLev-1)/2;
for ii=1:numMods
  for varNum=1:length(params(ii).mcmc.svars)
    svar=params(ii).mcmc.svars{varNum};
    svarS=params(ii).mcmc.svarSize(varNum);
    base.(svar)=2*ones(nLev,svarS);
    acc.(svar)=zeros(nLev,svarS);
  end
  % specialize in some cases
  for varNum=1:length(params(ii).mcmc.svars)
    var=params(ii).mcmc.svars{varNum};		
    switch(var)
    case 'theta'
      base.(var)(ex>0,:)=20.0^(2.0/(nLev-1));
    case {'betaV','betaU'}
      base.(var)(ex>0,:)=10.0^(2.0/(nLev-1));
    case {'lamUz','lamOs'}
      base.(var)(ex>0,:)=100.0^(2.0/(nLev-1));
    end
  end
  step(ii).base=base; 
  step(ii).ex=ex;
end

if ~isempty(hierParams)
 base.mean=2*ones(nLev); base.mean(ex>0)=20.0^(2.0/(nLev-1));
 base.lam=2*ones(nLev);
end

% pre-compute the widths for the levels, making whole mcmc structs to 
% substitute in and out of the params as we go through the levels 
for ii=1:numMods
 smcmc(ii,1:nLev)=params(ii).mcmc;
 for lev=1:nLev
  for varNum=1:length(params(ii).mcmc.svars)
    wvar=params(ii).mcmc.wvars{varNum};
    svar=params(ii).mcmc.svars{varNum};
    svarS=params(ii).mcmc.svarSize(varNum);
    for k=1:svarS
      smcmc(ii,lev).(wvar)(k)=smcmc(ii,lev).(wvar)(k)* ...
            step(ii).base.(svar)(lev,k)^step(ii).ex(lev);
      wrec(ii).(wvar)(k,lev)=smcmc(ii,lev).(wvar)(k);
    end
  end
 end
end

for hi=1:numHMods
 smcmcH(hi,1:nLev)=hierParams(hi).mcmc;
 for lev=1:nLev
  smcmcH(hi,lev).meanWidth=smcmcH(hi,lev).meanWidth* ...
         base.mean(lev)^ex(lev);
  wrecH(hi).meanWidth(lev)=smcmcH(hi,lev).meanWidth;
  smcmcH(hi,lev).lamWidth=smcmcH(hi,lev).lamWidth* ...
         base.lam(lev)^ex(lev);
  wrecH(hi).lamWidth(lev)=smcmcH(hi,lev).lamWidth;
  smcmcH(hi,lev).lockstepMeanWidth=smcmcH(hi,lev).lockstepMeanWidth* ...
         base.mean(lev)^ex(lev);
  wrecH(hi).lockstepMeanWidth(lev)=smcmcH(hi,lev).lockstepMeanWidth;
 end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Herein goeth the sampling code.  

fprintf('Collecting stepsize acceptance stats ...\n');

%init the params structs
[params hierParams]=gpmmcmc(params,0,'initOnly',1,'clist',clist,...
                            'hierParams',hierParams);

fprintf('Drawing %d samples (nBurn) over %d levels (nLev) \n',nBurn,nLev);
counter('stime',1,nBurn*nLev,15,10);
for burn=1:nBurn
  for lev=1:nLev
    counter((burn-1)*nLev+1+lev);
    for ii=1:numMods
      params(ii).mcmc=smcmc(ii,lev);
    end
    for hi=1:numHMods
      hierParams(hi).mcmc=smcmcH(hi,lev);
    end
    [params hierParams]=gpmmcmc(params,1,'noInit',1,'noCounter',1,...
                         'step',1,'clist',clist,'hierParams',hierParams);
  end
end
counter('end');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Estimate the optimal step size

fprintf('Computing optimal step sizes ...\n');

for ii=1:numMods
  for varNum=1:length(params(ii).mcmc.svars)
    svar=params(ii).mcmc.svars{varNum};
    svarS=params(ii).mcmc.svarSize(varNum);
    acc=[params(ii).pvals.([svar 'Acc'])]';
    accCount(ii).(svar)=zeros(nLev,svarS);
    for k=1:svarS
      accCount(ii).(svar)(:,k)=sum(reshape(acc(:,k),nLev,nBurn),2);
    end
  end
end

for hi=1:numHMods
  acc=[hierParams(hi).pvals.acc]';
  accCountH(hi).mean=sum(reshape(acc(:,1),nLev,nBurn),2);
  accCountH(hi).lam=sum(reshape(acc(:,2),nLev,nBurn),2);
  accCountH(hi).lockstepMean=sum(reshape(acc(:,3),nLev,nBurn),2);
end

nTrials=ones(nLev,1)*nBurn;
logit = log(1/(exp(1)-1));
for ii=1:numMods
  for varNum=1:length(params(ii).mcmc.svars)
    wvar=params(ii).mcmc.wvars{varNum};
    svar=params(ii).mcmc.svars{varNum};
    svarS=params(ii).mcmc.svarSize(varNum);
    switch(svar)
    case {'theta'}
      for k=1:svarS
        thisVarLinks=clist(clist(:,ii)==k,:);
        if any(thisVarLinks) & ~all(thisVarLinks(1:ii-1)==0)
          stepWidth(ii).(wvar)(k)=0;
        else
          widths=wrec(ii).(wvar)(k,:);
          b=glmfit(log(widths),[accCount(ii).(svar)(:,k) nTrials],'binomial');
          stepWidth(ii).(wvar)(k)=exp((logit-b(1))/b(2));
        end
      end
    otherwise
      for k=1:svarS
        widths=wrec(ii).(wvar)(k,:);
        b=glmfit(log(widths),[accCount(ii).(svar)(:,k) nTrials],'binomial');
        stepWidth(ii).(wvar)(k)=exp((logit-b(1))/b(2));
      end
    end
  end
end

for hi=1:numHMods
  widths=wrecH(hi).meanWidth; 
  b=glmfit(log(widths),[accCountH(hi).mean nTrials],'binomial');
  stepWidthH(hi).meanWidth=exp((logit-b(1))/b(2));
  widths=wrecH(hi).lamWidth;
  b=glmfit(log(widths),[accCountH(hi).lam nTrials],'binomial');
  stepWidthH(hi).lamWidth=exp((logit-b(1))/b(2));
  widths=wrecH(hi).lockstepMeanWidth;
  b=glmfit(log(widths),[accCountH(hi).lockstepMean nTrials],'binomial');
  stepWidthH(hi).lockstepMeanWidth=exp((logit-b(1))/b(2));
end

%put the estimated step sizes back into the params struct
for ii=1:numMods
  for varNum=1:length(params(ii).mcmc.svars)
    wvar=params(ii).mcmc.wvars{varNum};
    params(ii).mcmc.(wvar)=stepWidth(ii).(wvar);
  end
end

for hi=1:numHMods
  hierParams(hi).mcmc.meanWidth=stepWidthH(hi).meanWidth;
  hierParams(hi).mcmc.lamWidth=stepWidthH(hi).lamWidth;
  hierParams(hi).mcmc.lockstepMeanWidth=stepWidthH(hi).lockstepMeanWidth;
end

fprintf('Step size assignment complete.\n');

%params.mcmc.acc=accCount;
%params.mcmc.wrec=wrec;
%params.mcmc.smcmc=smcmc;
