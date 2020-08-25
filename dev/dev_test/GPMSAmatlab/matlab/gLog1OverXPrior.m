%function model = gLog1OverXPrior(x,parms)
%
% Computes log 1/x propr,
% sum of 1D distributions for each (x,parms) in the input vectors
% for use in prior likelihood calculation
%  parms = [native-scale-min native-scale-range]
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = gLog1OverXPrior(x,params)

x=x(:);

p = sum( - log( x.*params(:,2) + params(:,1) ) );

