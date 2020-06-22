function icVals = gGMICDF(means,vars,cVals)
% function icVals = gGMICDF(means,vars,Cvals)
% compute inverse CDF of a gaussian mixture(s)
% each row of means and vars defines a mixture
% output icVals is (rows of means&vars) by (length of cVals)

  icVals=zeros(size(means,1),length(cVals));
  sds=sqrt(vars);

  for ii=1:size(means,1)
    mingrid=min(means(ii,:)-4*sds(ii,:));
    maxgrid=max(means(ii,:)+4*sds(ii,:));
    grid=linspace(mingrid,maxgrid,1e4);
    fullMix=zeros(size(means,2),length(grid));
    for jj=1:size(means,2);
      fullMix(jj,:)=gNormpdf(grid,means(ii,jj),sds(ii,jj));
    end
    mm=sum(fullMix);
    icVals(ii,:)=empiricalICDF(grid,mm,cVals);
  end

    

end

function icdf=empiricalICDF(grid,updf,cdfVals)
  ecdf=cumsum(updf)/sum(updf);
  icdf=zeros(size(cdfVals));
  for ii=1:length(cdfVals)
    cLoc=find(ecdf>cdfVals(ii),1);
    if isempty(cLoc) 
      icdf(ii)=grid(end);
    elseif cLoc==1
      icdf(ii)=grid(1);
    else
      icdf(ii)=interp1(ecdf([cLoc-1 cLoc]),grid([cLoc-1,cLoc]),cdfVals(ii));
    end
  end
end
      