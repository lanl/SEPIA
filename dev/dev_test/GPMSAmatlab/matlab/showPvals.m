% function showPvals(pvals, skip)
%   skip = the beginning index to display; optional

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function showPvals(pvals, skip)

if ~exist('skip'); skip = 1; end;

fprintf('Processing pval struct from index %d to %d\n',skip,length(pvals));

f=fieldnames(pvals); fdel=false(length(f),1);
for ii=1:length(f) 
  if ~isempty(strfind(f{ii},'Acc'));  % don't show acc rate stats
    fdel(ii)=1; 
  end; 
  if ~isempty(strfind(f{ii},'logLik'));  % don't show likelihood (still post)
    fdel(ii)=1; 
  end; 
  if ~isempty(strfind(f{ii},'logPrior'));  % don't show prior (still post)
    fdel(ii)=1; 
  end; 
end
f=f(~fdel);
flen=length(f);

x=skip:length(pvals);
pvals=pvals(skip:end);

cla
for ii=1:flen
     y=[pvals.(f{ii})];
     h(ii)=pvalSubplot(flen,ii);
     if length(x)==length(y)
       plot(x,y);
     else
       plot(y);
     end
     ylabel(f{ii},'fontsize',10);
     if ~mod(ii,2); h(ii).YAxisLocation='right'; end
     fprintf('%10s:   mean           s.d. \n',f{ii})
     for jj=1:size(y,1)
       fprintf('       %3d: %12.4g %12.4g \n', ...
               jj,mean(y(jj,:)),std(y(jj,:)) );
     end
end
set(h(1:end-1),'XTick',[]);

end


% An internal function is needed to get the subplots to use more of the
% available figure space

function h=pvalSubplot(n,i) 

sep=0.25;
left  =0.1;
sizelr=0.8;
bottom=(1-(i/n))*0.8+0.1;
sizetb=(1/n)*0.8*(1-sep/2);

h=subplot('position',[left bottom sizelr sizetb]);
end
