%function model = computeLogPrior(priors,mcmc,model)
%
% Builds the prior likelihood
%

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
