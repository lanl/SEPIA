function rnorm = gMvnrnd(mu,cov,n)
% function rnorm = gMvnrnd(mu,cov,n)
% multivariante randoms, SVD method of solution
% mu=mean vector
% cov=covariance matrix
% n=number of randoms to draw (default=1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=mu(:);

  if ~exist('n'); n=1; end
  
  [U S V] = svd(cov);
  rnorm = repmat(mu,1,n) + U*sqrt(S) * randn(size(mu,1),n);
 
  rnorm=rnorm';

end 
