function h=gPackSubplot(n,m,i,j,sep);
%function h=gPackSubplot(n,m,i,j,sep);
% creates a subplot (axis) on an n,m grid at location i,j,
% where n and i are offsets from the top and
%       m and j are offsets from the left side
% If j is zero, i will be treated as a 1D index into the axes
% sep, if specified, is the border size as a proportion of the frame

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if j==0
  [j i]=ind2sub([m n],i);
end

if ~exist('sep'); sep=0; end

left  =((j-1)/m)*0.8+0.1;
sizelr=(1/m)*0.8*(1-sep/2);
bottom=(1-(i/n))*0.8+0.1;
sizetb=(1/n)*0.8*(1-sep/2);

h=subplot('position',[left bottom sizelr sizetb]);
