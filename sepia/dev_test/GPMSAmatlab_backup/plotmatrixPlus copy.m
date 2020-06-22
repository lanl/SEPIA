function plotmatrixPlus(data,varargin)
%function plotmatrixPlus(data,varargin)
% like plotmatrix, but with some extra functionality
% first argument is a dataset, rows are observations, columns are variables
% varargs:
%  labels: cell array of variable names
%  points: matrix of points to over-plot; columns are variables
%  xmin, xmax: vector of minimum and maximum values for axis scales.
%              (default = dataset min and max)
%  colorcode: color-codes data points in the lower triangle, varying color from 
%             blue (low value) to red (high value). Actually does 10 color
%             groups (rather than potentially #observations colors)
%    modes:
%      not specified: no colorcoding (data in blue, points in green)
%      1 -> colorcode data in row order (e.g. to reveal sequence)
%      vector of #observations: colorcode according to this tag. 
%  ucontour: put contour plots in the upper triangle (default=false)
%  lam: smoothess factor for the 2d kernel smoother (see ksContour)
%  ncpts: num points to use for kernel smoother (default 5000 - slow)

  [n,p]=size(data);

  points=[];
  labels=cell(1,p);
  xmin=min(data); xmax=max(data);
  colorcode=false;
  ucontour=false;
  lam=3;
  ncpts=5000;
  truepar=[];
  
  parseAssignVarargs({'labels','points','xmin','xmax', ...
                      'colorcode','ucontour','lam','ncpts','truepar'});

  if colorcode
    if length(colorcode)==1; colorcode=1:n; end 
    colorcode=colorcode(:);
    cp=prctile(colorcode,[0:10:100]);
    cvec=zeros(10,3);
    csel=zeros(n,1);
    for cc=1:10
      cvec(cc,:)=colorMorph([0 0 1],[1 0 0],(cc-1)/10);
      csel(colorcode>=cp(cc) & colorcode<=cp(cc+1))=cc;
    end
  end

  clf

  if ucontour; pmsamp=randsample(n,min(ncpts,n)); end

  [zH,zAX,zBigAx,zP,zPAx] = plotmatrix(data);
  for ii=1:length(zPAx)
     axi=[xmin(ii) xmax(ii)];
     a=axis(zPAx(ii)); 
     axis(zPAx(ii),[axi a(3:4) ]); 
     for jj=1:length(zPAx)
       if (colorcode) && (ii>jj)
         axes(zAX(ii,jj)); cla; hold on;
         for pp=1:10;
           plot(data(csel==pp,jj),data(csel==pp,ii),'.','color',cvec(pp,:))
         end
       end
       if (ucontour) && (ii~=jj)
         axes(zAX(ii,jj));
         ksContour(data(pmsamp,[jj ii]),[0.025 0.25 0.5 0.75 0.975], ...
               'ngrid',40,'ax',[xmin(jj) xmax(jj) xmin(ii) xmax(ii)], ...
               'lam',lam,'flip',false,'truepar',[truepar(jj),truepar(ii)])
       end
       axj=[xmin(jj) xmax(jj)];
       axis(zAX(ii,jj),[axj axi]);
       
       %alternate axes left/right and top/bottom
       if ~mod(ii,2) && (jj==p)
         set(zAX(ii,jj),'yAxisLocation','right')
       elseif mod(ii,2) && (jj==1)
       elseif ~mod(ii,2) && (jj==1)
         set(zAX(ii,jj),'yticklabel',{' '}); 
       else
         set(zAX(ii,jj),'ytick',[]);
       end
       if (ii==1); 
         set(zAX(ii,jj),'xAxisLocation','top');
         if ~mod(jj,2)
           set(zAX(ii,jj),'xticklabel',{' '}); 
         end
       elseif mod(jj,2) && (ii==p)
       else
         set(zAX(ii,jj),'xtick',[]); 
       end
       if exist('labels')
         if jj==1; ylabel(zAX(ii,jj),labels{ii}); end
         if ii==1; title(zAX(ii,jj),labels{jj}); end
       end
       hold(zAX(ii,jj),'on');  
       for kk=1:size(points,1)
         plot(zAX(ii,jj),points(kk,jj),points(kk,ii),'g.');
       end
         
    end
  end

    


end