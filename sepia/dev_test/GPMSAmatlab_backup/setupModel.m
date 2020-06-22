% function params=setupModel(obsData,simData,optParms)
% Sets up a gpmsa runnable struct from raw data.
% Please refer to associated documentation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
%
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
%
% Copyright 2008.  Los Alamos National Security, LLC. This material 
% was produced under U.S. Government contract DE-AC52-06NA25396 for 
% Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
% National Security, LLC for the U.S. Department of Energy. The U.S. 
% Government has rights to use, reproduce, and distribute this software.  
% NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY 
% WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF 
% THIS SOFTWARE.  If software is modified to produce derivative works, 
% such modified software should be clearly marked, so as not to confuse 
% it with the version available from LANL.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function params=setupModel(obsData,simData,optParms,varargin)

  verbose=1;
  parseAssignVarargs({'verbose'});

  % a shortcut version
    if isfield(obsData,'obsData') && isfield(obsData,'simData') && ...
       ~exist('optParms') 
      simData=obsData.simData; obsData=obsData.obsData;
    end
    
  % params is optional
    if ~exist('optParms'); optParms=[]; end 

  % check for and set up categorical data indicator/structs
    data.catInd=[];
    if isfield(optParms,'catInd')
      data.catInd=optParms.catInd;
    end

  % check for scalar output
    if isfield(optParms,'scalarOutput'); scOut=optParms.scalarOutput;
                                  else scOut=0; end
    model.scOut=scOut;

  % grab some parms to local (short named) vars
    n=length(obsData);
	  if n==0;  % eta-only model 
      % dummy up some empty fields
 	    obsData(1).x=[]; 
  		obsData(1).Dobs=[];
	  	obsData(1).yStd=[];
	    % all vars are x, otherwise some are x some are theta
      if iscell(simData.x)
        % separable design
        p=0;
        for ii=1:length(simData.x); p=p+size(simData.x{ii},2); end
      else
        p=size(simData.x,2);
      end
	    q=0;
    else
      p=length(obsData(1).x);
      if iscell(simData.x)
        % separable design
        q=-p;
        for ii=1:length(simData.x); q=q+size(simData.x{ii},2); end
      else
        q=size(simData.x,2)-p;
      end
    end
  	m=size(simData.yStd,2);
	  pu=size(simData.Ksim,2);
    pv=size(obsData(1).Dobs,2);

  if verbose
    fprintf('SetupModel: Determined data sizes as follows: \n')
    if n==0
      fprintf('SetupModel: This is a simulator (eta) -only model\n');
      fprintf('SetupModel: m=%3d  (number of simulated data)\n',m);
      fprintf('SetupModel: p=%3d  (number of inputs)\n',p);
      fprintf('SetupModel: pu=%3d (transformed response dimension)\n',pu);
    else
      fprintf('SetupModel: n=%3d  (number of observed data)\n',n);
      fprintf('SetupModel: m=%3d  (number of simulated data)\n',m);
      fprintf('SetupModel: p=%3d  (number of parameters known for observations)\n',p);
      fprintf('SetupModel: q=%3d  (number of additional simulation inputs (to calibrate))\n',q);
      fprintf('SetupModel: pu=%3d (response dimension (transformed))\n',pu);
      fprintf('SetupModel: pv=%3d (discrepancy dimension (transformed))\n',pv);
    end
    if iscell(simData.x)
      fprintf('SetupModel: Kronecker separable design specified\n');
    end
    if ~isempty(data.catInd)
      fprintf('SetupModel: categorical fields in positions ');
      fprintf('%d ',find(data.catInd));
      fprintf('\n');
    end
    fprintf('\n');
  end
  
  % check for and process lamVzGroups
    if isfield(optParms,'lamVzGroup'); lamVzGroup=optParms.lamVzGroup;
                                  else lamVzGroup=ones(pv,1); end
    lamVzGnum=length(unique(lamVzGroup));
    if ~isempty(setxor(unique(lamVzGroup),1:lamVzGnum))
      error('SetupModel: invalid lamVzGroup specification');
    end

  % put in a Sigy param if not supplied (backward compatability)
    if ~isfield(obsData,'Sigy')
      for k=1:n; obsData(k).Sigy=eye(size(obsData(k).Kobs,1)); end
    end
  % make a local copy of Lamy for use in this routine (do inv() only once)
    for k=1:n; obs(k).Lamy=inv(obsData(k).Sigy); end

  % Construct the transformed obs
    if scOut
       data.x=[]; data.u=[];
       for k=1:n;
         data.x(k,:)=obsData(k).x;
         data.u(k)=obsData(k).yStd;
       end;
    else
      % ridge to be used for stabilization
       DKridge=eye(pu+pv)*1e-6;

       data.x=[]; data.v=[]; data.u=[];
       for k=1:n;
         if (p>0); data.x(k,:)=obsData(k).x; end
        % Transform the obs data
         DK=[obsData(k).Dobs obsData(k).Kobs];
         vu=inv( DK'*obs(k).Lamy*DK + DKridge )* ...
            DK'*obs(k).Lamy*obsData(k).yStd;    
         data.v(k,:)=vu(1:pv);
         data.u(k,:)=vu(pv+1:end)';
       end;
    end
    if iscell(simData.x)  % add a composed  zt to the struct
      data.ztSep=simData.x;
      tdes=simData.x{end}; if size(tdes,1)==1; tdes=tdes'; end
      for ii=length(simData.x)-1:-1:1
        ndes=simData.x{ii}; if size(ndes,1)==1; ndes=ndes'; end
        [r1,r2]=meshgrid(1:size(simData.x{ii},1),1:size(tdes,1));
        tdes=[ndes(r1(:),:) tdes(r2(:),:)];
      end
      data.zt=tdes;
    else
      data.zt=simData.x;
      data.ztSep=[];
    end
    data.w=(simData.Ksim\simData.yStd)'; % Construct the transformed sim

  % check on the categorical indicator
  if ~isempty(data.catInd)
    if length(data.catInd)~=(p+q)
      error('SetupModel: categorical indicator not the same size as parameter vectors');
    end
    % assumption is the categories are integers in (0,ncat)
    tempxzt=[data.x zeros(n,q); data.zt];
    cchk=max(tempxzt);
    for ii=1:length(data.catInd)
      if data.catInd(ii)>0 && data.catInd(ii)~=cchk(ii); 
        error(['SetupModel: category limit #' num2str(ii) ' inconsistent with data']); 
      end
    end
  end

    
  % Set initial parameter values
    model.theta=0.5*ones(1,q);           % Estimated calibration variable
    model.betaV=ones(p,lamVzGnum)*0.1;   % Spatial dependence for V discrep
    model.lamVz=ones(lamVzGnum,1)*20;    % Marginal discrepancy precision
    model.betaU=ones(p+q,pu)*0.1;        % Sim PC surface spatial dependence
    model.lamUz=ones(pu,1)*1;            % Marginal precision
    model.lamWs=ones(pu,1)*1000;         % Simulator data precision
  % if there are categoricals, the theta values need to be positive integer
    if (~isempty(data.catInd))
      model.theta(data.catInd(p+1:end)~=0)=1;
    end

% Set up partial results to be stored and passed around;
  % Sizes, for reference:
    model.n=n; model.m=m; model.p=p; model.q=q;
    model.pu=pu; model.pv=pv;
    model.lamVzGnum=lamVzGnum; model.lamVzGroup=lamVzGroup;
  % Precomputable data forms and covariograms.
    model.x0Dist=genDist(data.x,data.catInd);
    model.ztDist=genDist(data.zt,data.catInd);
    if iscell(data.ztSep)  % then compute components of separable design
      for ii=1:length(data.ztSep)
        model.ztSepDist{ii}=genDist(data.ztSep{ii},data.catInd);
      end
    end
    model.w=data.w(:);
    if scOut
       model.uw=[data.u(:);data.w(:)];
       model.u=data.u(:);
    else
       model.vuw=[data.v(:);data.u(:);data.w(:)];
       model.vu=[data.v(:);data.u(:)];
    end

  % compute the PC loadings corrections
    model.LamSim=diag(simData.Ksim'*simData.Ksim);
    
  % initialize the acceptance record field
    model.acc=1;

  % compute LamObs, the u/v spatial correlation
    if scOut
      LO = zeros(n*pu);
      for kk=1:n
         ivals = (1:pu)+(kk-1)*pu;
         LO(ivals,ivals) = obs(kk).Lamy;
      end
      rankLO = rank(LO);
    else
      LO = zeros(n*(pv+pu));
      for kk=1:n
        DK = [obsData(kk).Dobs obsData(kk).Kobs];
        ivals = (1:pv+pu)+(kk-1)*(pv+pu);
        LO(ivals,ivals) = DK'*obs(kk).Lamy*DK;
      end
      rankLO = rank(LO);
      for kk=1:n
        ivals = (1:pv+pu)+(kk-1)*(pv+pu);
        LO(ivals,ivals) = LO(ivals,ivals) + DKridge;
      end
      % now reindex LamObs so that it has the v's first and the
      % u's 2nd.  LamObs is n*(pu+pv) in size and indexed in
      % the order v1 u1 v2 u2 ... vn un.  We want to arrange the
      % order to be v1 v2 ... vn u1 u2 ... un.  
      inew = [];
      for kk=1:pv
        inew = [inew; (kk:(pu+pv):n*(pu+pv))'];
      end
      for kk=1:pu
        inew = [inew; ((pv+kk):(pu+pv):n*(pu+pv))'];
      end
      LO = LO(inew,inew);
    end
    % compute the Penrose inverse of LO
    model.SigObs=inv(LO + 1e-8*eye(size(LO,1)) );
    
  % Set prior distribution types and parameters
    priors.lamVz.fname ='gLogGammaPrior';  
        priors.lamVz.params=repmat([1 0.0010],lamVzGnum,1);  
    priors.lamUz.fname ='gLogGammaPrior';  
        priors.lamUz.params=repmat([5 5],pu,1);  
    priors.lamWOs.fname='gLogGammaPrior';  
        priors.lamWOs.params=[5 0.005];  
    priors.lamWs.fname ='gLogGammaPrior';  
        priors.lamWs.params=repmat([3 0.003],pu,1);  
    priors.lamOs.fname ='gLogGammaPrior';  
        priors.lamOs.params=[1 0.001];  
    priors.rhoU.fname  ='gLogBetaPrior';   
        priors.rhoU.params=repmat([1 0.1],pu*(p+q),1);  
    priors.rhoV.fname  ='gLogBetaPrior';   
        priors.rhoV.params=repmat([1 0.1],p*lamVzGnum);  
    priors.theta.fname ='gLogNormalPrior'; 
        priors.theta.params=repmat([0.5 10],q,1);  

  % Modification of lamOs and lamWOs prior distributions
    if isfield(optParms,'priors')
       if isfield(optParms.priors,'lamWOs')
          priors.lamWOs.params=optParms.priors.lamWOs.params;
       end
       if isfield(optParms.priors,'lamOs')
          priors.lamOs.params=optParms.priors.lamOs.params;
       end
    end

  % Prior correction for lamOs and lamWOs prior values (due to D,K basis xform)
    %for lamOs, need DK basis correction
    totElements=0; 
    for ii=1:length(obsData); 
      totElements=totElements+length(obsData(ii).yStd);
    end
    aCorr=0.5*(totElements-rankLO);

    bCorr=0;
    if ~scOut
       for ii=1:n
         DKii = [obsData(ii).Dobs   obsData(ii).Kobs];
         vuii = [data.v(ii,:)'; data.u(ii,:)'];
         resid=obsData(ii).yStd(:) - DKii*vuii;  
         bCorr=bCorr+0.5*sum(resid'*obs(ii).Lamy*resid);
       end
    end
    priors.lamOs.params(:,1)=priors.lamOs.params(:,1)+aCorr;
    priors.lamOs.params(:,2)=priors.lamOs.params(:,2)+bCorr;
    
    %for lamWOs, need K basis correction
    aCorr=0.5*(size(simData.yStd,1)-pu)*m;
    ysimStdHat = simData.Ksim*data.w';
    bCorr=0.5*sum(sum((simData.yStd-ysimStdHat).^2));

    priors.lamWOs.params(:,1)=priors.lamWOs.params(:,1)+aCorr;
    priors.lamWOs.params(:,2)=priors.lamWOs.params(:,2)+bCorr;

  % Set the initial values of lamOs and lamWOs based on the priors.
    model.lamWOs=max(100,priors.lamWOs.params(:,1)/priors.lamWOs.params(:,2));
    model.lamOs=max(20, priors.lamOs.params(:,1)/priors.lamOs.params(:,2));

  % Set prior bounds 
    priors.lamVz.bLower=0;      priors.lamVz.bUpper=Inf;
    priors.lamUz.bLower=0.3;    priors.lamUz.bUpper=Inf;
    priors.lamWs.bLower=60;     priors.lamWs.bUpper=1e5;
    priors.lamWOs.bLower=60;    priors.lamWOs.bUpper=1e5;
    priors.lamOs.bLower=0;      priors.lamOs.bUpper=Inf;
    priors.betaU.bLower=0;      priors.betaU.bUpper=Inf;
    priors.betaV.bLower=0;      priors.betaV.bUpper=Inf;
    priors.theta.bLower=0;      priors.theta.bUpper=1;

    % Check if lamWOs outside bounds, reset and print message
    if model.lamWOs >= priors.lamWOs.bUpper
        fprintf('Warning: lamWOs initialized outside default bounds of [60, 1e5]; setting to 1e5 - 1.')
        model.lamWOs = 1e5 - 1;
    end

    % if thetaConstraintFunction supplied, use that, otherwise
    % use a dummy constraint function
      if isfield(optParms,'thetaConstraints')
        priors.theta.constraints=optParms.thetaConstraints;
        % update with the supplied initial theta
        model.theta=optParms.thetaInit;
        %ii=0;
        %while (ii<1e6) && ~tryConstraints(priors.theta.constraints,model.theta)
        %  model.theta=rand(size(model.theta));
        %  ii=ii+1;
        %end
        %if ii==1e6; error('unable to draw theta within constraints'); end
      else
        priors.theta.constraints={};
      end

      function constraintsOK=tryConstraints(constraints,theta)
        constraintsOK=1;
        for const=constraints
          constraintsOK=constraintsOK & eval(const{1});
        end
      end
      
  % Set mcmc step interval values
    mcmc.thetawidth=0.2 * ones(1,numel(model.theta));
    mcmc.rhoUwidth=0.1* ones(1,numel(model.betaU));
    mcmc.rhoVwidth=0.1* ones(1,numel(model.betaV));
    mcmc.lamVzwidth=10* ones(1,numel(model.lamVz));
    mcmc.lamUzwidth=5* ones(1,numel(model.lamUz));
    mcmc.lamWswidth=100* ones(1,numel(model.lamWs));
    mcmc.lamWOswidth=100* ones(1,numel(model.lamWOs));
    mcmc.lamOswidth=model.lamOs/2* ones(size(model.lamOs));
  % set up control var lists for sampling and logging
    % pvars is the list of variables from model struct to log
    % svars is the list of variables to sample (and compute prior on)
    % svarSize is the length of each svar variable
    % wvars is the list of corresponding mcmc width names
    if n>0 % if there's obsData, do the full deal.
     if pv>0
  	  mcmc.pvars={'theta','betaV','betaU','lamVz','lamUz','lamWs', ...
                  'lamWOs','lamOs','logLik','logPrior','logPost'};
	    mcmc.svars={'theta','betaV','betaU','lamVz', ...
	                'lamUz','lamWs','lamWOs','lamOs'};
   	  mcmc.svarSize=[q              % theta
                     p*lamVzGnum    % betaV
                     pu*(p+q)       % betaU
				  	         lamVzGnum      % lamVz
					           pu             % lamUz
					           pu             % lamWs
					           1              % lamWOs
					           1]';           % lamOs

  	  mcmc.wvars={'thetawidth','rhoVwidth','rhoUwidth','lamVzwidth', ...
  	              'lamUzwidth','lamWswidth','lamWOswidth','lamOswidth'};
     else %this is a no-discrepancy model with observations
  	  mcmc.pvars={'theta','betaU','lamUz','lamWs', ...
                  'lamWOs','lamOs','logLik','logPrior','logPost'};
	    mcmc.svars={'theta','betaU','lamUz','lamWs','lamWOs','lamOs'};
   	  mcmc.svarSize=[q              %theta
                     pu*(p+q)       % betaU
					           pu             % lamUz
					           pu             % lamWs
					           1           % lamWOs
                               1]';           % lamOs
	    mcmc.wvars={'thetawidth','rhoUwidth', ...
	                'lamUzwidth','lamWswidth','lamWOswidth','lamOswidth'};
       
     end
    else % we're doing just an eta model, so a subset of the params.
  	  mcmc.pvars={'betaU','lamUz','lamWs', ...
                  'lamWOs','logLik','logPrior','logPost'};
	    mcmc.svars={'betaU','lamUz','lamWs','lamWOs'};
   	  mcmc.svarSize=[pu*(p+q)       % betaU
					           pu             % lamUz
					           pu             % lamWs
					           1]';           % lamWOs
	    mcmc.wvars={'rhoUwidth', ...
	                'lamUzwidth','lamWswidth','lamWOswidth'};
    end
    
% Over and out
  params.data=data;
  params.model=model;
  params.priors=priors;
  params.mcmc=mcmc;
  params.obsData=obsData;
  params.simData =simData;
  params.optParms=optParms;
  params.pvals=[];  % initialize the struct

end
