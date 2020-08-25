function [h bigAx]=gPlotMatrix(data,varargin)
% function [h bigAx]=gPlotMatrix(data,varargin)
%  data - contains vectors for scatterplots
%    each row is an vector, as expected for plotmatrix
% varargs include
%  'Pcontours' are the percentile levels for the contour plot
%  'numBins' is the number of bins for diagonal histograms
%  'ngrid' is axis grid size (symmetric) (a good guess is 25, default=10)
%  'labels', a cell array of variable names [optional]
%  'ttl', an overall plot title [optional]
%  'axRange', a 2-vector of axis scalings, default [0,1] or data range
%  'ksd', the sd of the contour smoothing kernel (default=0.05)
%  'Pcontours', probability contours, default [0.5 0.9]
%  'ustyle', 'lstyle' is the type of the off-diagonal plots
%     'scatter' is xy scatterplots [default]
%     'imcont' is a smoothed image (2d est. pdf) with contours
%  'shade' causes the scatterplots to to shade from blue to red over
%     the input sequence of points
%  'marksize' is the MarkerSize argument to plot for scatterplots
%  'XTickDes' and 'YTickDes', if specified, are double cell arrays, containing 
%     pairs of designators. Designator {[0.5 0.75], {'1','blue'}} puts the
%     labels '1' and 'blue' at 0.5 and 0.75 on the pane, resp. The
%     outer cell array is length the number of axes.
%  'oneCellOnly' indicates that only one cell will be picked out, the cell
%     designated, e.g., [1 2]
%  'plotPoints' is a matrix of points to over-plot on scatterplots of images,
%     it has the same variables as the matrix being plotted
%  'plotPointsDes' is a plot designator for plotpoints, it's a cell array,
%     for example {'r*'} or {'r*','markersize',10}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  [n p]=size(data);

  % defaults for varargs
    labels=[];
    if any((data(:)<0)|any(data(:)>1));
      axRange=[min(data)' max(data)'];
    else
      axRange=repmat([0 1],p,1);
    end
    ksd=0.05; ngrid=10; Pcontours=[0.5;0.9];
    ustyle='scatter'; lstyle='scatter'; shade=0; ttl=[];  marksize=6;
    XTickDes=[]; YTickDes=[]; oneCellOnly=0;
    plotPoints=[]; plotPointsDes={'r*'};
    ldata=[];
    numBins=10;
    parseAssignVarargs({'labels','axRange','ngrid','ksd','Pcontours', ...
                        'ustyle','lstyle','shade','ttl','marksize', ...
                        'XTickDes','YTickDes','oneCellOnly', ...
                        'plotPoints','plotPointsDes','ldata','numBins'});  
    histldata=1;
    if isempty(ldata); ldata=data; histldata=0; end

  Pcontours=Pcontours(:);                      
  ncont=length(Pcontours);
  
  % if shading is enabled, set up the shade structs. 
   if shade
    shgroups=min(n,100); % need to set up groups if n is large
    sls=linspace(1,n,shgroups+1)'; slc=linspace(0,1,shgroups);
    for shi=1:shgroups; % define a range and a color for each group
      sh(shi).ix=ceil(sls(shi)):floor(sls(shi+1)); 
      sh(shi).color=(1-slc(shi))*[0 0 1] + slc(shi)*[1 0 0];
    end
   else
    shgroups=1; sh.ix=1:n; sh.color=[0 0 1];
   end
  
  % Put the data into the specified range
  %  (scale data to [0 1], where the axes will be set below)
    data=(data-repmat(axRange(:,1)',n,1)) ./ ...
              repmat((axRange(:,2)-axRange(:,1))',n,1);  
    if ~isempty(plotPoints)
      ppn=size(plotPoints,1);
      plotPoints=(plotPoints - repmat(axRange(:,1)',ppn,1)) ./ ...
              repmat((axRange(:,2)-axRange(:,1))',ppn,1);
    end
            
  % Generate a grid and supporting data structures
    gridvals = linspace(0,1,ngrid);
    [g1 g2] = meshgrid(gridvals,gridvals);
    g1v = g1(:); g2v = g2(:);
    gvlen = length(g1v);
    dens = zeros(gvlen,1);

  % begin
    clf;
    
  % establish the subplots
    for ii=1:p; for jj=1:p; 
      h(ii,jj)=gPackSubplot(p,p,ii,jj);
    end; end
  
  % Put in the histograms
    for ii=1:p
      axes(h(ii,ii));
      if ~histldata % single hist on diag
        hist(data(:,ii),numBins); 
      else  % two datasets; overlay kernel smooths
        for kk=1:length(gridvals)
          hdens(kk)=sum(calcNormpdf(data(:,ii),gridvals(kk),ksd));
        end
        plot(gridvals,hdens);
        hold on;
        for kk=1:length(gridvals)
          hdens(kk)=sum(calcNormpdf(ldata(:,ii),gridvals(kk),ksd));
        end
        plot(gridvals,hdens,'r');
      end
      %axisNorm(h(ii,ii),'x',[0 1]);
    end
  
    
   % Go through the 2D plots
   for ii=1:p-1; for jj=ii+1:p
     % compute the smooth and contours, if it's called for either triangle
     if any(strcmp({ustyle,lstyle},'imcont'))
       % compute the smooth response
         for i=1:gvlen
           f = calcNormpdf(data(:,jj),g1v(i),ksd) ...
             .*calcNormpdf(data(:,ii),g2v(i),ksd);
           dens(i) = sum(f);
         end
       % normalize dens
         dens = dens/sum(dens);
       % get the contours
         for j=1:ncont
          hlevels(j) = fzero(@(x) getp(x,dens)-Pcontours(j),[0 max(dens)]);
         end
       % precompute for data in lower triangle
         % compute the smooth response
           for i=1:gvlen
             f = calcNormpdf(ldata(:,jj),g1v(i),ksd) ...
               .*calcNormpdf(ldata(:,ii),g2v(i),ksd);
             ldens(i) = sum(f);
           end
         % normalize dens
           ldens = ldens/sum(ldens);
         % get the contours
           for j=1:ncont
            lhlevels(j) = fzero(@(x) getp(x,ldens)-Pcontours(j),[0 max(ldens)]);
           end
     end

     % Do the upper triangle plots
       axes(h(ii,jj));
       switch ustyle
       case 'scatter'
         for shi=1:shgroups
           plot(data(sh(shi).ix,jj),data(sh(shi).ix,ii),'.', ...
                'MarkerSize',marksize,'Color',sh(shi).color);
           hold on;
         end
       case 'imcont'
         imagesc(g1v,g2v,reshape(dens,ngrid,ngrid)); axis xy; hold on;
         colormap(repmat([.9:-.02:.3]',[1 3]));
         contour(g1,g2,reshape(dens,ngrid,ngrid),hlevels,'LineWidth',1.0,'Color','b'); 
       otherwise
         error('bad specification for lstyle');
       end
       if ~isempty(plotPoints)
          plot(plotPoints(:,jj),plotPoints(:,ii),plotPointsDes{:});
       end
       axis([0 1 0 1]);

     % Do the lower triangle plots
       axes(h(jj,ii)); 
       switch lstyle
       case 'scatter' 
         for shi=1:shgroups
           plot(ldata(sh(shi).ix,ii),ldata(sh(shi).ix,jj),'.', ...
                'MarkerSize',marksize,'Color',sh(shi).color);
           hold on;
         end
         hold on;
       case 'imcont'
         imagesc(g1v,g2v,reshape(ldens,ngrid,ngrid)'); axis xy; hold on;
         colormap(repmat([.9:-.02:.3]',[1 3]));
         contour(g1,g2,reshape(ldens,ngrid,ngrid)',lhlevels,'LineWidth',1.0,'Color','b'); 
       otherwise
         error('bad specification for lstyle');
       end
       if ~isempty(plotPoints)
          plot(plotPoints(:,ii),plotPoints(:,jj),plotPointsDes{:});
       end
       axis([0 1 0 1]);
     
   end; end

   % Ticks and Tick labels, by default they're not there
   set(h,'XTick',[],'YTick',[]);
   % but put them on if specified.
   if ~isempty(XTickDes)
     for ii=1:size(h,2)
        set(h(end,ii),'XTick',XTickDes{ii}{1});
        set(h(end,ii),'XTickLabel',XTickDes{ii}{2});
     end
   end
   if ~isempty(YTickDes)
     for ii=1:size(h,1)
          set(h(ii,1),'YTick',YTickDes{ii}{1});
          set(h(ii,1),'YTickLabel',YTickDes{ii}{2});
     end
   end

   % labels
   if ~isempty(labels)
     for ii=1:p
       %title(h(1,ii),labels{ii});
       ylabel(h(ii,1),labels{ii});
       xlabel(h(end,ii),labels{ii});
     end
   end
   
   % if a title was supplied, put it up relative to an invisible axes
    if ~isempty(ttl)
      bigAx=axes('position',[0.1 0.1 0.8 0.8],'visible','off'); hold on;
      text(0.5,1.05,ttl,'horizontalalignment','center','fontsize',14);
    end 

   if oneCellOnly
     set(h(oneCellOnly(1),oneCellOnly(2)), ...
         'position',[0.075 0.075 0.85 0.85]); 
   end
    
end


% function to get probability of a given level h
function pout = getp(h,d);
    iabove = (d >= h);
    pout = sum(d(iabove));
end

function y=calcNormpdf(x,m,s)
  %calculate a multivariate normal pdf value
  n=size(s);
  nf=1./( sqrt(2*pi) .* s );
  up=exp(-0.5* ((x-m)./s).^2);
  y=nf.*up;
end
