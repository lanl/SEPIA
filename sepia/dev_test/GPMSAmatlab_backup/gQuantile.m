function qvals=gQuantile(xin,qs,dim)
% function qvals=gQuantile(x,qs)
% substitute for statistics toolbox quantile function, by Gatt
% returns the quantiles specified in qs for each column in x

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

if exist('quantile')==2; % use the builtin if available
  if exist('dim'); qvals=quantile(xin,qs,dim);
              else qvals=quantile(xin,qs);
  end
  return
end

if any(qs<0) || any(qs>1)
  error('gQuantile accepts quantile spec''d in the range [0,1]');
end

if ~exist('dim'); 
  dim=1;
  if isvector(xin); xin=xin(:); end % turn it to a col vec
end

p=length(qs);

perm=[dim setxor(1:max(ndims(xin),dim),dim)];
xperm=permute(xin,perm);

sz=size(xperm);
n=sz(1); m=prod(sz(2:end));
x=reshape(xperm,n,m);

qvals=zeros(p,m);

for jj=1:m
  col=sort(x(:,jj));
  col=col([1 1:end end]);
  inds=[0 ((1:n)-0.5)/n 1];
  qvals(:,jj)=interp1(inds,col,qs,'linear');
end

qvals=reshape(qvals,[p sz(2:end)]);

qvals=ipermute(qvals,perm);


