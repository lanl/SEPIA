function qvals=gQuantile(xin,qs,dim)
% function qvals=gQuantile(x,qs)
% substitute for statistics toolbox quantile function, by Gatt
% returns the quantiles specified in qs for each column in x

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
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


