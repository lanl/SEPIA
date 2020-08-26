%function model = computeLogPrior(priors,mcmc,model)
%
% Builds the prior likelihood
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model = computeLogPrior(priors,mcmc,model)


f=mcmc.svars;

logprior=0;
lastp=0;

for ii=1:length(f);
  curf=f{ii};

  switch(curf)
    % betaU ad betaV have to be handled with rho/beta transformation
    case 'betaU'
      rhoU= exp(-model.betaU.*(0.5^2));
      rhoU(rhoU>0.999)=0.999;
      logprior=logprior + feval(priors.rhoU.fname,rhoU,priors.rhoU.params);
    case 'betaV'
      rhoV= exp(-model.betaV.*(0.5^2));
      rhoV(rhoV>0.999)=0.999;
      logprior=logprior + feval(priors.rhoV.fname,rhoV,priors.rhoV.params);
    otherwise % it's general case for the others
      logprior=logprior + ...
         feval(priors.(curf).fname,model.(curf),priors.(curf).params);
  end
   %fprintf('%10s %f\n',curf,logprior-lastp); lastp=logprior;
   %fprintf('%10s %f\n',curf,logprior);
end

model.logPrior=logprior;
