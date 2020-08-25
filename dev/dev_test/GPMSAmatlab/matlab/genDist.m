% function d = gendist(data,dataDesc);
%   generates the nxnxp distance array values and supporting
%   information, given the nxp location matrix x
%   or if a d is passed in, just update the distances

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = genDist(data,catInd)

d.type=1;

[n,p] = size(data);

if ~exist('catInd','var')
  warning('categorical Indicator not passed to genDist');
  catInd=[];
end
if ~isempty(catInd); catInd=catInd(1:p); end

%generate the list of (n-1)*(n-1) distance indices
  inds=n*(n-1)/2;
  indi=zeros(inds,1);indj=zeros(inds,1);
  ind=1;for ii=1:n-1; indi(ind:ind+n-ii-1)=ii; indj(ind:ind+n-ii-1)=ii+1:n; ind=ind+n-ii; end;
 
  d.n=n; d.p=p;
  d.indi=indi; d.indj=indj;
  d.indm=indi + n*(indj-1);

if p==0; d.d=[]; return; end

if isempty(catInd)
  d.d=(data(indj,:)-data(indi,:)).^2;
else
  d.d=zeros(inds,p);
  isCat=(catInd~=0);
  d.d(:,~isCat)=(data(indj,~isCat)- data(indi,~isCat)).^2;
  d.d(:, isCat)=(data(indj, isCat)~=data(indi, isCat))*0.5;
end

end  
  