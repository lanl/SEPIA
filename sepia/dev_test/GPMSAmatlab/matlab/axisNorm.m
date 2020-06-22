% function axisNorm(handles, mode, axisVals)
% Tool to set 2S plot axes to the same values.
%   handles is a list of handles to the plots in question
%   mode is combinations of 'x', 'y', and 'z', optionally followed by 'max'
%     indicating which axes are to be set, and whether they are to be
%     autoscaled to the outer bounds of all, or to the given values
%     in axisVals.
%      For example
%        'xmax' scales the x axis in all handles to the max bounds;
%        'xyzmax' scales all axes to their max enclosures
%        'xy' scales the x and y axes to values in axisVals
%  axisVals, if supplied, has dummy values in unspecified positions
% 'imrange' mode normalizes the range of the images (that is,
%   the CLim axis properties)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function axisNorm(h, mode, ax) 
%test
h=h(:);

if strcmp(mode,'imrange')
  clim=[inf -inf];
  for ii=1:length(h);
     cl=get(h(ii),'CLim');
     clim(1)=min(clim(1),cl(1)); clim(2)=max(clim(2),cl(2));
  end
  set(h,'CLim',clim);
  return
end

maxMode=0; xParm=0; yParm=0; zParm=0;
if regexp(mode,'max');
   maxMode=1; 
   mode=mode(1:regexp(mode,'max')-1);
end
if regexp(mode,'x'); xParm=1; end
if regexp(mode,'y'); yParm=1; end
if regexp(mode,'z'); zParm=1; end

if maxMode  % then determine the enclosing axes
  axNum=length(axis(h(1)));
  axMult=repmat([-1 1],1,axNum/2);
  ax=-Inf*ones(1,axNum);
  for ii=1:length(h)
    ax=max([ax; axis(h(ii)).*axMult]);
  end
  ax=ax.*axMult;
  mode=mode(1:regexp(mode,'max')-1);
end  

for ii=1:length(h)
  a=axis(h(ii));
  if xParm
    a([1 2])=ax([1 2]); 
  end
  if yParm
    a([3 4])=ax([3 4]); 
  end
  if zParm
    a([5 6])=ax([5 6]); 
  end
  axis(h(ii),a);
end
