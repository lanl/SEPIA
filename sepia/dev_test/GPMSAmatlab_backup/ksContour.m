function ksContour(data,Pcontours,varargin)
%  data - contains 2D points to use to estimate pdf contours.
%  Pcontours are the percentile (of prob mass) levels for the contour plot
%  optional arguments name/value list:
%    ngrid - grid size for ks estimation, default = 25
%    ax - (axes) if supplied, specifies the range of the grid. 
%         default: +/- 10% data range
%    numPtsAdd: select this many points from data, randomly, to overplot.
%    lam: smoothing coefficients, see ksdensity2d
%    flip: flip the plot u-d (for plotmatrix upper triangle)

  ngrid=50; %ceil(sqrt(size(data,1)));
  numPtsAdd=0;
  drange=range(data);
  ax(1)=min(data(:,1))-drange(1)*0.1;
  ax(2)=max(data(:,1))+drange(1)*0.1;
  ax(3)=min(data(:,2))-drange(2)*0.1;
  ax(4)=max(data(:,2))+drange(2)*0.1;
  lam=1;
  flip=false;
  truepar=[];

  parseAssignVarargs({'ngrid','ax','numPtsAdd','lam','flip','truepar'});
  
  gridx=linspace(ax(1),ax(2),ngrid); gridy=linspace(ax(3),ax(4),ngrid);

  z=ksdensity2d(data,gridx,gridy,lam,lam);
  if flip; z=flipud(z); end
  
  z=z/sum(z(:));
  zSort=sort(z(:));
  zCDF=cumsum(zSort);
  for ii=1:length(Pcontours)
    cVals(ii)=zSort(find(zCDF>Pcontours(ii),1));
  end
  cla;
  image(gridx,gridy,(1-0.4*repmat(z/max(z(:)),[1,1,3])).^3);
  hold on;
  contour(gridx,gridy,z,cVals);
  axis(ax);
  
  %add true parameter values
    if ~isempty(truepar)
    %hold on; 
       plot(truepar(1),truepar(2), 'sk', 'MarkerSize', 7,'MarkerFaceColor','r');
    end
  
  % add data points
  if numPtsAdd>0
    n=size(data,1);
    gs=gSample(n,numPtsAdd);
    hold on;
    plot(data(gs,1),data(gs,2),'.');
  end
  
end