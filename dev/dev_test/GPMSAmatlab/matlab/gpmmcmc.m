% function [params hierParams] = gpmmcmc(params,nmcmc,varargin)
%  params - a parameters struct or array of structs
%  nmcmc - number of full draws to perform (overridden for step mode)
%  varargs are in string/value pairs
%    'noCounter' - default 0, 1 ==> do not output a counter of iterations
%    'step'      - default 0, 1 ==> specified step size mode
%                     (presumes optimized step sizes for lambda parameters,
%                      rather than calculating adaptive step sizes)
%    'initOnly'  - only do & return the precomputaton of partial results
%    'noInit'    - initialization is not necessary, skip precomputation
%    'clist'     - for multiple models, the description of common thetas
%                  each row is for one linked variable (theta). A row is a
%                  list of indicators the same length as the number of 
%                  models (the params array). A zero indicates the 
%                  corresponding theta is not in the corresponding model, a
%                  nonzero entry indicates the index of that theta in that 
%                  model.
%    'hierParams'- parameter structure for hierarchical model linking of
%                  theta parameters in joint models. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
%         Brian Williams, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [params,hierParams] = gpmmcmc(params,nmcmc,varargin)

% Grab to local variable
  numMods=length(params);

% Process input arguments
  noCounter=0; clist=zeros(0,numMods);
  step=0; initOnly=0; noInit=0;
  hierParams=[];
  parseAssignVarargs({'noCounter', 'clist','hierParams', ...
                      'step','initOnly','noInit'});

 % Backwards compatibility 
  for modi=1:numMods
   % for zt as a single field 
   if ~isfield(params(modi).data,'zt'); 
        params(modi).data.zt=[params(modi).data.z params(modi).data.t]; 
   end
   % for separable design field (indicator)
   if ~isfield(params(modi).data,'ztSep'); params(modi).data.ztSep=[]; end
   % backwards compat for categorical capability
   if ~isfield(params(modi).data,'catInd'); params(modi).data.catInd=[]; end
  end
   
 % if there is a hierarchical model, seed the models' priors. 
   params=copyHPriors(params,hierParams);
     
% Initialize the models 
 if ~noInit
   for modi=1:numMods
    % computing the likelihood sets up partial results inside the model structure
      C.var='all';
      params(modi).model=computeLogLik(params(modi).model,params(modi).data,C); 
      params(modi).model=computeLogPrior(params(modi).priors,params(modi).mcmc,...
                                       params(modi).model);
      params(modi).model.logPost=params(modi).model.logPrior+params(modi).model.logLik;
   end 
 end
 
if initOnly; return; end
 
 % initialize the structure that will record draw info
 for modi=1:numMods
   if numel(params(modi).pvals);
     pvals=params(modi).pvals; poff=length(pvals);
   else
     for var=params(modi).mcmc.pvars; params(modi).pvals(1).(var{1})=0; end
     for var=params(modi).mcmc.svars; params(modi).pvals(1).([var{1} 'Acc'])=0; end; 
     poff=0;
   end
   params(modi).pvals(poff+nmcmc)=params(modi).pvals(1);
 end

% pull out the minimal subset data structure to pass around
  for modi=1:numMods;
    subParams(modi).model=params(modi).model;
    subParams(modi).data=params(modi).data;
    subParams(modi).priors=params(modi).priors;
    subParams(modi).mcmc=params(modi).mcmc;
  end
  
% Counter will be used and displayed if we are not in linked model mode
  if ~noCounter; counter('stime',1,nmcmc,10,10); end

    % Do mcmc draws
    for iter=1:nmcmc
      if ~noCounter; counter(iter); end
            
      for modi=1:numMods
        % Get some local vars picked out
          svars=subParams(modi).mcmc.svars; 
          svarSize=subParams(modi).mcmc.svarSize;
          wvars=subParams(modi).mcmc.wvars; 

        for varNum=1:length(svars)
          C.var=svars{varNum};
          C.aCorr=1; % default is no step correction.
          switch(C.var)
          case {'theta'}
            for k=1:svarSize(varNum)
              C.index=k;
              % Check for a categorical parameter
              catParam=0;
              if (~isempty(subParams(modi).data.catInd))
                catParam=subParams(modi).data.catInd(subParams(modi).model.p+k);
              end
              if ~catParam
                C.val=subParams(modi).model.(C.var)(k) + ...
                      (rand(1)-.5)*subParams(modi).mcmc.(wvars{varNum})(k);
              else % it's categorical, draw a random category.
                C.val=1;
                if catParam>1  % check for non-degenerate category
                  curVal=subParams(modi).model.(C.var)(k);
                  C.val=floor(randi(catParam-1));
                  if (C.val>=curVal); C.val=C.val+1; end % to not select current
                  %fprintf('Cat draw, was %d, drew %d  |  ',curVal,C.val);
                end
              end
              subParams=mcmcEval(subParams,modi,C,clist);
              acc.(C.var)(k)=subParams(modi).model.acc;
            end
          case {'betaV','betaU'}
            for k=1:svarSize(varNum)
              cand = exp(-subParams(modi).model.(svars{varNum})(k).*(.5^2))+ ...
                         (rand(1)-.5)*subParams(modi).mcmc.(wvars{varNum})(k);
              C.index=k;C.val=-log(cand)/(0.5^2);
              subParams=mcmcEval(subParams,modi,C,clist);
              acc.(C.var)(k)=subParams(modi).model.acc;
            end
          case {'lamVz','lamUz','lamWs','lamWOs','lamOs'}
            for k=1:svarSize(varNum)
              if ~step
                C.index=k;
                [C.val, C.aCorr]=chooseVal(subParams(modi).model.(C.var)(k));
                subParams(modi).model.acc=0; %might not call eval
                if C.aCorr; 
                  subParams=mcmcEval(subParams,modi,C,clist); 
                end
              else
                C.index=k;C.val=subParams(modi).model.(C.var)(k) + ...
                              (rand(1)-.5)*subParams(modi).mcmc.(wvars{varNum})(k);
                subParams=mcmcEval(subParams,modi,C,clist);
              end
              acc.(C.var)(k)=subParams(modi).model.acc;
            end
          otherwise  
            error('Unknown sample variable in gpmmcmc mcmcStep')
          end          

        end

        % Save the designated fields
        for varName=subParams(modi).mcmc.pvars
          params(modi).pvals(poff+iter).(varName{1})=...
            subParams(modi).model.(varName{1})(:);
        end
        for varName=subParams(modi).mcmc.svars
          params(modi).pvals(poff+iter).([varName{1} 'Acc'])=acc.(varName{1})(:);
        end
       
      end % going through the models
            
      % if there is a hierarchical models structure, perform sampling
      if ~isempty(hierParams)
        [subParams,hierParams] = mcmcHier(subParams,hierParams,step,poff+iter);

        for hi=1:length(hierParams);
          for vi=1:length(hierParams(hi).vars)
            modi=hierParams(hi).vars(vi).modNum;
            varNum=hierParams(hi).vars(vi).varNum;
            if hierParams(hi).pvals(poff+iter).acc(3)
              params(modi).pvals(poff+iter).theta(varNum)=...
              subParams(modi).model.theta(varNum);
            end
          end
        end
      end
      
    end % going through the iterations
    
  if ~noCounter; counter('end'); end % end the counter

% recover the main data structure
  for modi=1:numMods;
    params(modi).model= subParams(modi).model;
    params(modi).data=  subParams(modi).data;
    params(modi).priors=subParams(modi).priors;
  end

% And that's it for the main function ....
end %main function gaspmcmc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [params hP] = mcmcHier(params,hP,step,logi)
pr=0;
oldPriorStyle=0;
% if there is a hierarchical model on thetas, this is where
% the hyperparameters are sampled and updated

% go through all hierarchical models specified
for hi=1:length(hP)
  reject=zeros(7,1);
  % First, sample the hyperparameters
  % generate a candidate draw and evaluate for mean
   cand=hP(hi).model.mean + (rand(1)-.5)*hP(hi).mcmc.meanWidth;
   if cand<hP(hi).priors.mean.bLower || cand>hP(hi).priors.mean.bUpper
     if pr; fprintf(' Reject, out of bounds 1\n'); end
     reject(1)=1;
   end
   if ~reject(1)
     constraintsOK=1;
     for jj=1:length(hP(hi).vars)
       modind=hP(hi).vars(jj).modNum; varNum=hP(hi).vars(jj).varNum;
       theta=params(modind).priors.theta.params(:,1); theta(varNum)=cand;
       for const=params(modind).priors.theta.constraints
         constraintsOK=constraintsOK & eval(const{1});
       end
     end
     if ~constraintsOK
       if pr
         fprintf(' Reject, from hierarchical mean constraint set 1\n');
       end
       reject(1)=1;
     end
   end
   if ~reject(1)
     [params hP reject(2)]=evalHierDraw(cand,hP(hi).model.lam,1,params,hP,hi);
   end

  % generate a candidate draw and evaluate for lam
   aCorr=1;
   if ~step
    [cand aCorr]=chooseVal(hP(hi).model.lam);
   else
    cand=hP(hi).model.lam+(rand(1)-.5)*hP(hi).mcmc.lamWidth;
   end
   if cand<hP(hi).priors.lam.bLower || cand>hP(hi).priors.lam.bUpper
     if pr; fprintf(' Reject, out of bounds 3\n'); end
     reject(3)=1;
   else
     [params hP reject(4)]=evalHierDraw(hP(hi).model.mean,cand,aCorr,params,hP,hi);

   end

  % Second, try a lockstep update of the hierarchical and individual
  % theta means (to avoid shrinkage overstability problems) This moves
  % all of the points and the h.model mean by the same shift, so the
  % only changes are the models' likelihoods and the hier mean prior
    candDelta=(rand(1)-.5)*hP(hi).mcmc.lockstepMeanWidth;
    % candidate for the hierarchical mean
      newHmean=hP(hi).model.mean+ candDelta;
      if pr
        fprintf('candDelta=%f, old=%f new=%f; ', ...
                candDelta, hP(hi).model.mean,newHmean); 
      end
    % Check bounds
      if newHmean < hP(hi).priors.mean.bLower || ...
         newHmean > hP(hi).priors.mean.bUpper
          if pr; fprintf(' Reject, out of bounds 5\n'); end
          reject(5)=1;
      end
      if ~reject(5)
         constraintsOK=1;
         for jj=1:length(hP(hi).vars)
           modind=hP(hi).vars(jj).modNum; varNum=hP(hi).vars(jj).varNum;
           theta=params(modind).priors.theta.params(:,1);
           theta(varNum)=newHmean;
           for const=params(modind).priors.theta.constraints
             constraintsOK=constraintsOK & eval(const{1});
           end
         end
         if ~constraintsOK
            if pr
              fprintf(' Reject, from hierarchical mean constraint set 5\n');
            end
            reject(5)=1;
         end
      end
    if ~reject(5)
      % compute the updated Normal prior for the hier. model mean
      if oldPriorStyle
        curHPrior=-0.5*( (hP(hi).model.mean-hP(hi).priors.mean.mean)./  ...
                        (hP(hi).priors.mean.std) ).^2;
        newHPrior=-0.5*( (newHmean-hP(hi).priors.mean.mean)./  ...
                        (hP(hi).priors.mean.std) ).^2;
      else
        curHPrior=gLogNormalPrior(hP(hi).model.mean, ...
                     [hP(hi).priors.mean.mean hP(hi).priors.mean.std]);
        newHPrior=gLogNormalPrior(newHmean, ...
                     [hP(hi).priors.mean.mean hP(hi).priors.mean.std]);
      end
      if pr
        fprintf('prior from %f to %f; ',curHPrior,newHPrior);
      end
      % Compute the updated likelihood for the associated models
      C.var='theta';
      for jj=1:length(hP(hi).vars)
        modind=hP(hi).vars(jj).modNum; varNum=hP(hi).vars(jj).varNum;
        modelT(jj)=params(modind).model;
        curLik(jj)=modelT(jj).logLik;
        modelT(jj).theta(varNum)=modelT(jj).theta(varNum)+candDelta;
        % check bounds for the model
          if modelT(jj).theta(varNum) < params(modind).priors.theta.bLower || ...
             modelT(jj).theta(varNum) > params(modind).priors.theta.bUpper
              if pr; fprintf(' Reject, out of bounds 6\n'); end
              reject(6)=1;
          end
          theta=modelT(jj).theta';
          constraintsOK=1;
          for const=params(modind).priors.theta.constraints
             constraintsOK=constraintsOK & eval(const{1});
          end
          if ~constraintsOK
             if pr; fprintf(' Reject, from theta constraint set 6\n'); end
             reject(6)=1;
          end
        if ~reject(6)
          % set the var and compute the new likelihood
          modelT(jj)=computeLogLik(modelT(jj),params(modind).data,C); 
          % extract the new likelihood value
          newLik(jj)=modelT(jj).logLik;
        end
      end
    end
    if ~any(reject(5:6))
      % Add up the priors and likelihoods
      oldPost=sum(curLik)+curHPrior;
      newPost=sum(newLik)+newHPrior;
      if pr
        fprintf('lik from %f to %f; ',sum(curLik),sum(newLik));
      end
      % Test acceptance and update
      if log(rand)<(newPost-oldPost)
        % record the new hier model mean
        hP(hi).model.mean=newHmean;
        for jj=1:length(hP(hi).vars)
          modind=hP(hi).vars(jj).modNum; varNum=hP(hi).vars(jj).varNum;
          % Update models, after adding up the correct posterior lik
          modelT(jj).logPost=modelT(jj).logLik+modelT(jj).logPrior;
          params(modind).model=modelT(jj);
          % Update model prior mean as the new hyper param mean
          params(modind).priors.theta.params(varNum,1)= newHmean;
        end
      else
        reject(7)=1;
      end
    end

  if pr; if ~any(reject); fprintf('accept \n'); 
            else fprintf('reject %d\n',find(reject)); end
  end

  hP(hi).pvals(logi).mean=hP(hi).model.mean;
  hP(hi).pvals(logi).lam=hP(hi).model.lam;
  hP(hi).pvals(logi).acc=[~any(reject(1:2)) ~any(reject(3:4)) ...
                          ~any(reject(5:7))]';
end
end % function mcmcHier

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluate whether to accept a change to hier priors.
function [params,hP,reject]=evalHierDraw(mean,lam,aCorr,params,hP,hi)
  oldPriorStyle=0;
    % go through each linked var, find out what the updated prior is
    for vari=1:length(hP(hi).vars)
      modind=hP(hi).vars(vari).modNum; varNum=hP(hi).vars(vari).varNum;
      curPrior(vari)=params(modind).model.logPrior;
      priorsT(vari)=params(modind).priors;
      priorsT(vari).theta.params(varNum,1)=mean; 
      priorsT(vari).theta.params(varNum,2)=sqrt(1/lam);
      modelT=computeLogPrior(priorsT(vari),...
                             params(hP(hi).vars(vari).modNum).mcmc, ...
                             params(hP(hi).vars(vari).modNum).model);
      newPrior(vari)=modelT.logPrior;
    end
    % compute the hierarchical prior parameter likelihoods
    % mean is a normal prior
    % lambda is a gamma prior
    if oldPriorStyle
      curHPrior=0.5* ( (hP(hi).model.mean-hP(hi).priors.mean.mean)./  ...
                       (hP(hi).priors.mean.std) ).^2;
      newHPrior=0.5* ( (mean-hP(hi).priors.mean.mean)./  ...
                       (hP(hi).priors.mean.std) ).^2;
      curHPrior=curHPrior+(hP(hi).priors.lam.a-1).*log(hP(hi).model.lam)- ...
                        hP(hi).priors.lam.b*hP(hi).model.lam;
      newHPrior=newHPrior+(hP(hi).priors.lam.a-1).*log(lam)- ...
                         hP(hi).priors.lam.b*lam;
    else
      curHPrior=gLogNormalPrior(hP(hi).model.mean, ...
                   [hP(hi).priors.mean.mean hP(hi).priors.mean.std]);
      newHPrior=gLogNormalPrior(mean, ...
                   [hP(hi).priors.mean.mean hP(hi).priors.mean.std]);
      curHPrior=curHPrior+gLogGammaPrior(hP(hi).model.lam, ...
                   [hP(hi).priors.lam.a hP(hi).priors.lam.b]);
      newHPrior=newHPrior+gLogGammaPrior(lam, ...
                   [hP(hi).priors.lam.a hP(hi).priors.lam.b]);
    end
    % sum up the priors
      oldLogPrior=sum(curPrior) + curHPrior;
      newLogPrior=sum(newPrior) + newHPrior;

    % check for acceptance
    if ( log(rand)<(newLogPrior-oldLogPrior + log(aCorr)) )
      reject=0;
      % accept! record the current vals
      hP(hi).model.mean=mean; hP(hi).model.lam=lam;
      % put stuff back into the submodel prior structs, update the
      % calculated prior and posterior lik.
      for vari=1:length(hP(hi).vars)
        modind=hP(hi).vars(vari).modNum;
        params(modind).priors=priorsT(vari);
        params(modind).model.logPrior=newPrior(vari);
        params(modind).model.logPost=newPrior(vari)+params(modind).model.logLik;
      end
    else
      reject=1;
    end
end % function EvalHierDraw

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function params=mcmcEval(params,modi,C,clist)

  params(modi).model.acc=0; % default status is to not accept

  model=params(modi).model; data=params(modi).data; 
  priors=params(modi).priors; mcmc=params(modi).mcmc;

  pr=0; % print diagnostics
  if pr; fprintf('%s %2d %.2f ',C.var,C.index,C.val); end
  if pr; fprintf(':: %.2f %.2f ',model.logLik,model.logPrior); end

  % If var is a theta, must check whether it is linked to other models
  thisVarLinks=[];
  if strcmp(C.var,'theta')
    thisVarLinks=clist(clist(:,modi)==C.index,:);
    if any(thisVarLinks) & ~all(thisVarLinks(1:modi-1)==0) 
      % only the first in round robin samples the linked variable
      if pr; fprintf('  Skipping linked var, index %d\n',C.index); end
      return
    end
  end

  % Check hard parameter bounds.
  if (C.val<priors.(C.var).bLower || ...
      priors.(C.var).bUpper<C.val || ...
      ~isreal(C.val));
    % if a theta and categorical ignore bounds violation (note the "not")
    if ~ ( strcmp(C.var,'theta') && ... 
           ~isempty(params(modi).data.catInd) && ...
           (params(modi).data.catInd(params(modi).model.p+C.index)~=0) )
      if pr; fprintf(' Reject, out of bounds\n'); end
      return
    end
  end

  modelT=model;
  modelT.(C.var)(C.index)=C.val;
  modelT=computeLogPrior(priors,mcmc,modelT);
  modelT=computeLogLik(modelT,data,C);
  modelT.logPost=modelT.logPrior+modelT.logLik;

  % if theta, consult the constraint function
  if strcmp(C.var,'theta')
    theta=modelT.theta';
    constraintsOK=1;
    for const=priors.theta.constraints
      constraintsOK=constraintsOK & eval(const{1});
    end
    if ~constraintsOK
      if pr; fprintf('  Reject, from theta constraint set\n'); end
      return 
    end
  end

  % If we are in a linked model, compute the likelihood correction
  lOldLik=[]; lNewLik=[];
  linkInds=find(thisVarLinks~=0); %all the links
  if linkInds; linkInds(find(linkInds==modi))=[]; end %this mod's doesn't count as a link
  for link=1:length(linkInds)
    lModelT(link)=params(linkInds(link)).model;
    lOldLik(link)=lModelT(link).logLik;
    lModelT(link).theta(thisVarLinks(linkInds(link)))=modelT.theta(C.index);
    D.var='theta';
    lModelT(link)=computeLogLik(lModelT(link),params(linkInds(link)).data,D);
    lNewLik(link)=lModelT(link).logLik;
    if pr;
      fprintf('\n  Linked vars; LL of model %d var %d from %8.5f to %8.5f', ...
        linkInds(link),thisVarLinks(linkInds(link)),lOldLik(link),lNewLik(link));
    end
  end
  if pr && ~isempty(linkInds); fprintf('\n'); end

  oldLogPost=model. logPost + sum(lOldLik);
  newLogPost=modelT.logPost + sum(lNewLik);

  if pr; fprintf(':: %.2f %.2f ',modelT.logLik,modelT.logPrior); end
  if pr && ~isempty(linkInds); 
    fprintf('\n links changge LL as old %f to new %f',oldLogPost,newLogPost);
  end

  %if strcmp(C.var,'theta')
  %  if ~isempty(params(modi).data.catInd) 
  %    if (params(modi).data.catInd(params(modi).model.p+C.index)~=0) 
  %       pr=1;
  %    end
  %  end
  %end
  
  if pr; fprintf('MCMC test: old %6.2f new %6.2f corr %6.2f ', ...
                 newLogPost,oldLogPost,log(C.aCorr)); end
  if ( log(rand)<(newLogPost-oldLogPost + log(C.aCorr)) )
    if pr; fprintf(' Accept \n'); end
    model=modelT;
    model.acc=1;
    % if we are in a linked model, update the linked theta vals & mods
    for link=1:length(linkInds)
      lModelT(link)=computeLogPrior(params(linkInds(link)).priors,...
                    params(linkInds(link)).mcmc,lModelT(link));
      lModelT(link).logPost=lModelT(link).logPrior+lModelT(link).logLik;
      params(linkInds(link)).model=lModelT(link);
      if pr; fprintf('updated linked model %d\n',linkInds(link)); end
    end

  else
    if pr; fprintf(' Reject \n'); end
  end

  params(modi).model=model;

end % function mcmcEval

%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dval,acorr]=chooseVal(cval)
  % select the interval, and draw a new value.
  w=max(1,cval/3);
  dval=cval + (rand*2-1)*w;
  % do a correction, which depends on the old and new interval
  w1=max(1,dval/3);
  if cval > (dval+w1)
    acorr=0;
  else
    acorr=w/w1;
  end
  % fprintf('cval=%10.4f, dval=%10.4f, acorr=%10.4f\n',cval,dval,acorr)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
function params=copyHPriors(params,hP)
  % if there is a hierarchical model, sync the models' priors. 
  for ii=1:length(hP)
    for jj=1:length(hP(ii).vars)
      modind=hP(ii).vars(jj).modNum; varNum=hP(ii).vars(jj).varNum;
      params(modind).priors.theta.params(varNum,1)=hP(ii).model.mean;
      params(modind).priors.theta.params(varNum,2) =sqrt(1/hP(ii).model.lam);
    end
  end
end

