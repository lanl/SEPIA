% This defines a hierarchical model parameter structure as an example. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function hierParams=setupDefaultHierParams

% Hier is a struct array, each element represents one hierarchical
% model that addresses variables from models in a joint model analysis

% the links define the variables. This example creates a hierarchical model
% with two variables, the first variable (theta) in models 1 and 2.
hierParams(1).vars(1).modNum=1;
hierParams(1).vars(1).varNum=1;
hierParams(1).vars(2).modNum=2;
hierParams(1).vars(2).varNum=1;

% a starting point and a stored location for the hierarchical model
% the hierarchical distribution is a normal, with mean and precision
hierParams(1).model.mean=0.5;
hierParams(1).model.lam=10;

% priors for the hierarchical parameters
% the mean is from a normal dist, the lam is from a gamma
hierParams(1).priors.mean.mean=0.5;
hierParams(1).priors.mean.std=10;
hierParams(1).priors.mean.bLower=0;
hierParams(1).priors.mean.bUpper=1;
hierParams(1).priors.lam.a=1;
hierParams(1).priors.lam.b=1e-8;
hierParams(1).priors.lam.bLower=0;
hierParams(1).priors.lam.bUpper=Inf;

% and a place for mcmc control parameters
  hierParams(1).mcmc.meanWidth=0.1;
  % lockstep update parameters
  hierParams(1).mcmc.lockstepMeanWidth=0.1;
  % lambda will be sampled as an adaptive parameter

% a place for recording the samples, in the pvals structure
hierParams(1).pvals.mean=[];
hierParams(1).pvals.lam=[];

% this is where you would put in the next hierParams struct array to cover
% further hierarchical models in the analysis

end
