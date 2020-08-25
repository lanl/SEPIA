function list=sample(len,num,w)
% function list=sample(len,num,weights)
% returns a list if num distinct indexes into a sample of size len
% weights, if supplied, causes the sample to be weighted by the weights W.
%    in this case, len is taken from the weights vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('w')
  [y i]=sort(rand(len,1));
  list=sort(i(1:num));
else
  wc=[0 cumsum(w(:)/sum(w))'];
  r=sort(rand(num,1));
  list=zeros(num,1);
  index=1;
  for ii=1:length(r);
    while wc(index)<r(ii); index=index+1;end
    list(ii)=index-1;
  end
  
end