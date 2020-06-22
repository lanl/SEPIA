function pp=gPredProcess(params,pred)
% function ppred=gPredProcess(params,pred)
% do some common processing on the predictions in pred:
%   5,10,20,50,80,90,95 percentile and mean of
%   PC loadings, scaled space, mean-zero space, and native space
% If it's a wpred, then we'll just do the simulation space predictions
% If it's a uvpred, then do simulation and discrepancy

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


% pull out like fields for processing
if isfield(pred,'u') & isfield(pred,'v')
  ep=pred.u;
  dp=pred.v; isdp=true;
elseif isfield(pred,'w')
  ep=pred.w;
  dp=[]; isdp=false;
else
  error('unrecognized pred structure in gPredProcess')
end

Ksim=params.simData.Ksim;
if isdp
  Kobs=params.obsData(1).Kobs;
  Dobs=params.obsData(1).Dobs;
  if isfield(params.simData,'Dsim')
    Dsim=params.simData.Dsim;
  elseif isfield(params.simData.orig,'Dsim')
    Dsim=params.simData.orig.Dsim;
  elseif isfield(params.obsData(1).orig,'Dsim')
    Dsim=params.obsData(1).orig.Dsim;
  elseif size(Kobs,1)==size(Ksim,1)
    Dsim=Dobs;
  else
    fprintf('Cannot obtain valid Dsim\n');
    Dsim=nan(size(Ksim,1),size(Dobs,2));
  end
end

% recast the pred mats to npred by nbasis
[p1 p2 p3]=size(ep); 
ep=permute(reshape(permute(ep,[2 1 3]),[p2 p1*p3]),[2 1]);
if isdp
  [p1 p2 p3]=size(dp);
  dp=permute(reshape(permute(dp,[2 1 3]),[p2 p1*p3]),[2 1]);
end

n=size(ep,1);
[xx ps]=size(params.simData.yStd');
pu=size(Ksim,2);
if isdp; pv=size(Dobs,2); end
[xx po]=size(params.obsData(1).yStd');

% sim grid mean and sd
  ysd=params.simData.orig.ysd(:); 
    if isscalar(ysd); ysd=repmat(ysd,ps,1); end
  ymean=params.simData.orig.ymean(:);

% it will go like this:
%pp.pc
%pp.scaled
%pp.mean0
%pp.native
%
%pp.obs.scaled.eta.pc5
%pp.sim.scaled.delta.mean

% PC space
  pp.pc.eta.pred=ep;
  pp.pc.eta.mean=mean(ep);
  pct=prctile(ep,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
    pp.pc.eta.prc5 =pct(1,:); 
    pp.pc.eta.prc10=pct(2,:); 
    pp.pc.eta.prc20=pct(3,:); 
    pp.pc.eta.prc50=pct(4,:); 
    pp.pc.eta.prc80=pct(5,:); 
    pp.pc.eta.prc90=pct(6,:); 
    pp.pc.eta.prc95=pct(7,:); 
  if isdp
    pp.pc.delta.pred=dp;
    pp.pc.delta.mean=mean(dp);
    pct=prctile(dp,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
      pp.pc.delta.prc5 =pct(1,:); 
      pp.pc.delta.prc10=pct(2,:); 
      pp.pc.delta.prc20=pct(3,:); 
      pp.pc.delta.prc50=pct(4,:); 
      pp.pc.delta.prc80=pct(5,:); 
      pp.pc.delta.prc90=pct(6,:); 
      pp.pc.delta.prc95=pct(7,:); 
  end
  
% Scaled space
  ep=ep*Ksim';
  pp.scaled.eta.pred=ep;
  pp.scaled.eta.mean=mean(ep);
  pct=prctile(ep,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
    pp.scaled.eta.prc5 =pct(1,:); 
    pp.scaled.eta.prc10=pct(2,:); 
    pp.scaled.eta.prc20=pct(3,:); 
    pp.scaled.eta.prc50=pct(4,:); 
    pp.scaled.eta.prc80=pct(5,:); 
    pp.scaled.eta.prc90=pct(6,:); 
    pp.scaled.eta.prc95=pct(7,:); 
  if isdp
    dp=dp*Dsim';
    pp.scaled.delta.pred=dp;
    pp.scaled.delta.mean=mean(dp);
    pct=prctile(dp,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
      pp.scaled.delta.prc5 =pct(1,:); 
      pp.scaled.delta.prc10=pct(2,:); 
      pp.scaled.delta.prc20=pct(3,:); 
      pp.scaled.delta.prc50=pct(4,:); 
      pp.scaled.delta.prc80=pct(5,:); 
      pp.scaled.delta.prc90=pct(6,:); 
      pp.scaled.delta.prc95=pct(7,:); 
  end
  
% mean0 space
  ep=ep.*repmat(ysd',n,1);
  pp.mean0.eta.pred=ep;
  pp.mean0.eta.mean=mean(ep);
  pct=prctile(ep,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
    pp.mean0.eta.prc5 =pct(1,:); 
    pp.mean0.eta.prc10=pct(2,:); 
    pp.mean0.eta.prc20=pct(3,:); 
    pp.mean0.eta.prc50=pct(4,:); 
    pp.mean0.eta.prc80=pct(5,:); 
    pp.mean0.eta.prc90=pct(6,:); 
    pp.mean0.eta.prc95=pct(7,:); 
  if isdp
    dp=dp.*repmat(ysd',n,1);
    pp.mean0.delta.pred=dp;
    pp.mean0.delta.mean=mean(dp);
    pct=prctile(dp,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
      pp.mean0.delta.prc5 =pct(1,:); 
      pp.mean0.delta.prc10=pct(2,:); 
      pp.mean0.delta.prc20=pct(3,:); 
      pp.mean0.delta.prc50=pct(4,:); 
      pp.mean0.delta.prc80=pct(5,:); 
      pp.mean0.delta.prc90=pct(6,:); 
      pp.mean0.delta.prc95=pct(7,:); 
  end
  
% native space
  ep=ep+repmat(ymean',n,1);
  pp.native.eta.pred=ep;
  pp.native.eta.mean=mean(ep);
  pct=prctile(ep,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
    pp.native.eta.prc5 =pct(1,:); 
    pp.native.eta.prc10=pct(2,:); 
    pp.native.eta.prc20=pct(3,:); 
    pp.native.eta.prc50=pct(4,:); 
    pp.native.eta.prc80=pct(5,:); 
    pp.native.eta.prc90=pct(6,:); 
    pp.native.eta.prc95=pct(7,:); 
  if isdp
    dp=dp; % delta stays the same
    pp.native.delta.pred=dp;
    pp.native.delta.mean=mean(dp);
    pct=prctile(dp,[5 10 20 50 80 90 95]); if isvector(pct); pct=pct(:); end
      pp.native.delta.prc5 =pct(1,:); 
      pp.native.delta.prc10=pct(2,:); 
      pp.native.delta.prc20=pct(3,:); 
      pp.native.delta.prc50=pct(4,:); 
      pp.native.delta.prc80=pct(5,:); 
      pp.native.delta.prc90=pct(6,:); 
      pp.native.delta.prc95=pct(7,:); 
  end
  


end