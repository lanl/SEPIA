%function model = gLogGammaPrior(x,parms)
%
% Computes unscaled log normal pdf,
% sum of 1D distributions for each (x,parms) in the input vectors
% for use in prior likelihood calculation
%  parms = [a-parameter-vector  b-parameter-vector]
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = gLogGammaPrior(x,parms)

a=parms(:,1); b=parms(:,2);
x=x(:);

p=sum( (a-1).*log(x) - b.*x );
