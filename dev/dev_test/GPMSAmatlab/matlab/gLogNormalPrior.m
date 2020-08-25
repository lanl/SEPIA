%function model = gLogNormalPrior(x,parms)
%
% Computes unscaled log normal pdf,
% sum of 1D distributions for each (x,parms) in the input vectors
% for use in prior likelihood calculation
%  parms = [mean-vector  standard-deviation-vector]
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = gLogNormalPrior(x,parms)

mu=parms(:,1); std=parms(:,2);
x=x(:);
p = - .5 * sum( ((x-mu)./std).^2 );
