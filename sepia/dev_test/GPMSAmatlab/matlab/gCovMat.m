% function Scov = gCovMat(dist,beta,lamz,lams)
% given n x p matrix x of spatial coords, and dependence parameters
% beta p x 1, this function returns a matrix built from the
% correlation function
%     Scov_ij = exp{- sum_k=1:p beta(k)*(x(i,k)-x(j,k))^2 } ./lamz

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Scov = gCovMat(dist,beta,lamz,lams)

% check for case of a null dataset
%if isempty(dist.d); Scov=[]; return; end
  
if dist.type==1
  n=dist.n;
  Scov=zeros(n);
  if n>0  % if it's not a null dataset, do the distances
    % Scov(dist.indm)=exp(-(dist.d*beta))./lamz; %%% this is faster: 
       t=exp(-(dist.d*beta))./lamz;
       Scov(dist.indm)=t;
    Scov=Scov+Scov';
    diagInds = 1:(n+1):(n*n);
    if nargin==4   % then a lams was passed in 
      Scov(diagInds)=1/lamz + 1/lams;
    else
      Scov(diagInds)=1/lamz;
    end
  end
elseif dist.type==2
  n=dist.n; m=dist.m;
  Scov=zeros(n,m);
  if n*m >0 % if it's not a null dataset, do the distances
    %Scov(dist.indm)=exp(-(dist.d*beta))./lamz; %%% this is faster: 
      t=exp(-(dist.d*beta))./lamz;
      Scov(dist.indm)=t;
  end
else  
  error('invalid distance matrix type in gaspcov');
end
