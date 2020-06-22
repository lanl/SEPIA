% function Scov = gCovMat(dist,beta,lamz,lams)
% given n x p matrix x of spatial coords, and dependence parameters
% beta p x 1, this function returns a matrix built from the
% correlation function
%     Scov_ij = exp{- sum_k=1:p beta(k)*(x(i,k)-x(j,k))^2 } ./lamz

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
