% function d = gendist2(data1,data2,dataDesc);
%   generates the nxmxp distance array values and supporting
%   information, given the nxp matrix data1 and mxp data2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function d = genDist2(data1,data2,catInd)

d.type=2;
[n,p1] = size(data1);
[m,p2] = size(data2);
p=max(p1,p2);

if ~exist('catInd')
  warning('categorical Indicator not passed to genDist');
  catInd=[];
end
if ~isempty(catInd); catInd=catInd(1:p); end

%generate & store the list of n*m distance indices
  inds=n*m;
  indi=repmat(1:n,1,m);
  indj=repmat(1:m,n,1); indj=indj(:)';
 
  d.n=n; d.m=m; d.p=p;
  d.indi=indi; d.indj=indj;
  d.indm=indi + n*(indj-1);

if any([p1 p2]==0); d.d=[]; return; end % if either dataset is empty

if isempty(catInd);
  d.d=(data1(indi,:)-data2(indj,:)).^2;
else
  d.d=zeros(inds,p);
  isCat=(catInd~=0);
  d.d(:,~isCat)=(data1(indi,~isCat)- data2(indj,~isCat)).^2;
  d.d(:, isCat)=(data1(indi, isCat)~=data2(indj, isCat))*0.5;
end
  