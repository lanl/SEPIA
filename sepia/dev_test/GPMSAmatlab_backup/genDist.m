% function d = gendist(data,dataDesc);
%   generates the nxnxp distance array values and supporting
%   information, given the nxp location matrix x
%   or if a d is passed in, just update the distances

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
  