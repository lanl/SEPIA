% function gBoxPlot(x,varargin)
% substitute for stats toolbox boxplot function, by Gatt
% shows a boxplot-like summary for each column of x
% lines of the box are at the lower quartile, median, and upper quartile
% whiskers extend to the most extreme values with 1.5 times the
% inter-quartile range,
% extreme values outside that are plotted as 'x'
% the only option currently implemented is 'labels' as cell array, i.e.:
%  gBoxPlot(rand(10,2),'labels',{'varname1','varname2'});
% May also request no outlier labels, with 'noOutliers' optional argument=1

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

function gBoxPlot(x,varargin)

labels=[];
noOutliers=0;
parseAssignVarargs({'labels','noOutliers'});

if exist('boxplot')==2 && ~noOutliers; % use the builtin if available
  if ~isempty(labels)
    boxplot(x,'labels',labels);
  else
    boxplot(x);
  end
  return
end

cla; hold on;
if min(size(x))==1; x=x(:); end
[m n]=size(x);
boxsize=[-1 1] * (0.05 + 0.2*(1-exp(-(n-1))));

for jj=1:size(x,2);
  col=x(:,jj);
  qs=gQuantile(col,[0.25 0.5 0.75]);
  plot(jj+boxsize,[1 1]*qs(1));
  plot(jj+boxsize,[1 1]*qs(2),'r');
  plot(jj+boxsize,[1 1]*qs(3));
  plot([1 1]*boxsize(1)+jj,qs([1 3]));
  plot([1 1]*boxsize(2)+jj,qs([1 3]));
  
  % establish whisker low and high limits
  xlr=(qs(1)-1.5*(qs(3)-qs(1)));
  xlh=(qs(3)+1.5*(qs(3)-qs(1)));

  %get the whisker levels & plot (nearest within limit)
  wlow=min(col(col>xlr));
  whig=max(col(col<xlh));
  plot(0.5*boxsize+jj,[1 1]*wlow,'k');
  plot([1 1]*jj,[wlow qs(1)],'k--');   
  plot(0.5*boxsize+jj,[1 1]*whig,'k');
  plot([1 1]*jj,[whig qs(3)],'k--');  
  
  if ~noOutliers
    %plot the remaining extremes
    xlow=col(col<xlr); 
    xhig=col(col>xlh);
    for ii=[xlow; xhig]';
      plot(jj,ii,'r+');
    end    
  end

end

xmin=min(x(:)); xmax=max(x(:)); xrange=xmax-xmin;
a=axis; axis([0.5 jj+0.5 xmin-0.05*xrange xmax+0.05*xrange]);

set(gca,'xtick',1:n);
if ~isempty(labels)
  set(gca,'xticklabel',labels);
else
  xlabel('Column number');
end
ylabel('Values');
