function dens = ksdensity2d(data,gridx,gridy,lamx,lamy)
%function dens = ksdensity2d(data,gridx,gridy,lamx,lamy)
% data is n by 2
% gridx, gridy are the (regular) grid over which the smoothing 
%     kernels will be applied
% lamx, lamy adjust kernel s.d., i.e. the smoothing. 
%     1=default, larger = smoother. 

  [g1,g2] = meshgrid(gridx,gridy);
  g1v = g1(:); g2v = g2(:);
  gvlen = length(g1v);
  if ~exist('lamx','var'); lamx=1; end
  if ~exist('lamy','var'); lamy=1; end
  ksdx=mean(diff(gridx))/2*lamx;
  ksdy=mean(diff(gridy))/2*lamy;

  dens=zeros(gvlen,1);
  for i=1:gvlen
    dens(i) =sum( normpdf(data(:,1),g1v(i),ksdx).*normpdf(data(:,2),g2v(i),ksdy) );
  end

  dens=reshape(dens/sum(dens),length(gridx),length(gridy));

if nargout==0
  % Plot the estimate
  surf(gridx,gridy,dens);
  hold on; 
  plot(data(:,1),data(:,2),'.')
  
end

end

