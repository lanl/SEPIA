% function model = computeLogLik(model,data,C)
% 
% Builds the log likelihood of the data given the model parameters.
% 

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

function model = computeLogLik(model,data,C)

 n=model.n;   m=model.m;
pu=model.pu; pv=model.pv;
 p=model.p;   q=model.q;
lamVzGnum=model.lamVzGnum; lamVzGroup=model.lamVzGroup;

% backwards compat for categorical capability
if ~isfield(data,'catInd'); data.catInd=[]; end

% validate and process the changed field
do.theta=0;do.betaV=0;do.lamVz=0;do.betaU=0;do.lamUz=0;
do.lamWs=0;do.lamWOs=0;

if strcmp(C.var,'theta') || strcmp(C.var,'all') % update distances
       model.xDist=genDist([data.x repmat(model.theta,n,1)],data.catInd);
       model.xzDist=genDist2([data.x repmat(model.theta,n,1)],data.zt,data.catInd);
end

switch C.var
    case 'all'
       do.theta=1; do.betaV=1; do.lamVz=1; do.betaU=1; 
       do.lamUz=1; do.lamWs=1; do.lamWOs=1; do.lamOs=1; 
       model.SigWl=zeros(pu,1);       
    case 'theta'; do.theta=1;
    case 'betaV'; do.betaV=1;
    case 'lamVz'; do.lamVz=1;
    case 'betaU'; do.betaU=1;
    case 'lamUz'; do.lamUz=1;
    case 'lamWs'; do.lamWs=1;
    case 'lamWOs'; do.lamWOs=1;
    case 'lamOs'; do.lamOs=1;
    otherwise
        %error('Invalid Subtype in computeLogLik');
end

betaV=model.betaV;   lamVz=model.lamVz;
betaU=model.betaU;   lamUz=model.lamUz;
lamWs=model.lamWs;   lamWOs=model.lamWOs; lamOs=model.lamOs;

% Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm

  if (do.theta || do.betaV || do.lamVz)
	SigV=[];
    for jj=1:lamVzGnum
      SigV(jj).mat=gCovMat(model.x0Dist, betaV(:,jj), lamVz(jj));
    end
    model.SigV=SigV;
  else
    SigV=model.SigV;
  end

  if (do.theta || do.betaU || do.lamUz || do.lamWs)
    SigU(pu).mat=[];
    diags1=diagInds(n);
    for jj=1:pu
        SigU(jj).mat=gCovMat(model.xDist,betaU(:,jj),lamUz(jj));
        SigU(jj).mat(diags1)=SigU(jj).mat(diags1)+1/lamWs(jj);
    end
    model.SigU=SigU;
  else
    SigU=model.SigU;
  end

  if (do.betaU || do.lamUz || do.lamWs || do.lamWOs)
    diags2=diagInds(m);
    switch C.var
      case 'all'
        jinds=1:pu;
      case 'betaU'
        jinds=ceil( C.index/(p+q) );
      case {'lamUz','lamWs'}
        jinds=C.index;
      case 'lamWOs'
        jinds=1:pu;
    end
    for jj=jinds
      if isempty(data.ztSep)
        cg=gCovMat(model.ztDist,betaU(:,jj),lamUz(jj));
        cg(diags2)=cg(diags2)+1/(model.LamSim(jj)*lamWOs) + 1/lamWs(jj);

        % calculate the SigW likelihood for each block
          model.SigWl(jj)=doLogLik(cg,model.w((jj-1)*m+1:jj*m));
        % calculate the SigW inverse for each block 
          if (n>0) % but it's only needed for a calibration model
            model.SigWi(jj).mat=inv(cg);
          end
      else
        % there is a separable design, so compute these as kron'ed blocks
        segVarStart=1;
        for ii=1:length(data.ztSep)
           segVars=segVarStart:(segVarStart + model.ztSepDist{ii}.p-1);
           segVarStart=segVarStart+ model.ztSepDist{ii}.p;
           if (ii==1)  % ensure that lamUz is only counted once
             cg{ii}=gCovMat(model.ztSepDist{ii},betaU(segVars,jj),lamUz(jj));
           else
             cg{ii}=gCovMat(model.ztSepDist{ii},betaU(segVars,jj),1);
           end
        end
        cgNugget=1/(model.LamSim(jj)*lamWOs) + 1/lamWs(jj);
        [model.SigWl(jj) model.V(jj).mats model.Dki2(jj).vec]= ...
            doLogLikSep(cg,cgNugget,model.w((jj-1)*m+1:jj*m));
      end
    end
  end

  % The computation is decomposed into the likelihood of W, 
  %  and the likelihood of VU|W. 

  % Compute the likelihood of the W part (have already done the blocks)
    LogLikW=sum(model.SigWl);

  % only if there are observations do we have the VU|W part. 
  if (n>0)  
    % Compute the likelihood of the VU|W
    if (do.theta || do.betaU || do.lamUz)
      SigUW(pu).mat=[];
      for jj=1:pu
          SigUW(jj).mat=gCovMat(model.xzDist,betaU(:,jj),lamUz(jj));
      end
      model.SigUW=SigUW;
    else
      SigUW=model.SigUW;
    end

    % do these ops, on the block diagonal blocks: 
    %    W=SigUW*model.SigWi;
    %    SigUgW=SigU-W*SigUW';
      W(pu).mat=[];
      SigUgW(pu).mat=[];
      for ii=1:pu
        if isempty(data.ztSep)
          W(ii).mat=SigUW(ii).mat*model.SigWi(ii).mat;
          SigUgW(ii).mat=SigU(ii).mat-W(ii).mat*SigUW(ii).mat';
        else
          % computation for a kron/separable design
          zp=zeros(size(SigUW(ii).mat'));
          zp2=zeros(size(zp));
          for kk1=1:(size(zp,2))
          %zp2=zp .* model.Dki2(ii).vec;
            zp(:,kk1)=sepQuadFormCalc(model.V(ii).mats,SigUW(ii).mat(kk1,:)');
            zp2(:,kk1)=zp(:,kk1).*model.Dki2(ii).vec;
          end
          SigUgW(ii).mat=SigU(ii).mat - zp2'*zp2;
          W(ii).mat=zp2;
        end
      end

    %for scalar output: SigVUgW=[SigV+SigUgW] ...
    %                          + model.SigObs/lamOs;
    %otherwise:         SigVUgW=[SigV             zeros(n*pv,n*pu); ...
    %                            zeros(n*pu,n*pv) SigUgW         ] ...
    %                          + model.SigObs/lamOs;
      SigVUgW=model.SigObs/lamOs;
        for ii=1:pv
          blkRange=(ii-1)*n+1:ii*n;
          SigVUgW(blkRange,blkRange)=SigVUgW(blkRange,blkRange)+ ...
                                                  SigV(lamVzGroup(ii)).mat;
        end
      if model.scOut
        for ii=1:pu
          blkRange=(ii-1)*n+1:ii*n;
          SigVUgW(blkRange,blkRange)=SigVUgW(blkRange,blkRange)+SigUgW(ii).mat;
        end
      else
        for ii=1:pu
          blkRange=n*pv+((ii-1)*n+1:ii*n);
          SigVUgW(blkRange,blkRange)=SigVUgW(blkRange,blkRange)+SigUgW(ii).mat;
        end
      end

    % do this op: MuVUgW =W*model.w;
      MuVUgW=zeros(n*pu,1);
      for ii=1:pu
        blkRange1=(ii-1)*n+1:ii*n;
        blkRange2=(ii-1)*m+1:ii*m;
        if isempty(data.ztSep)
          MuVUgW(blkRange1)=W(ii).mat*model.w(blkRange2);
        else
          % computation for a kron/separable design
          zp=sepQuadFormCalc(model.V(ii).mats,model.w(blkRange2));
            zp2=zp .* model.Dki2(ii).vec;
          MuVUgW(blkRange1)=W(ii).mat'*zp2;
        end
      end

    % for scalar output:  MuDiff=   [u] - [MuVUgW]
    % otherwise:          MuDiff= [v;u] - [0;MuVUgW] 
      if model.scOut
         MuDiff=model.u; MuDiff=MuDiff-MuVUgW;
      else
         MuDiff=model.vu; MuDiff(pv*n+1:end)=MuDiff(pv*n+1:end)-MuVUgW;
      end

    % Now we can get the LL of VU|W
      LogLikVUgW=doLogLik(SigVUgW,MuDiff);

  else
    LogLikVUgW=0;
  end % test on whether we have observations (n>0)
  
  % Final Answer, LL(VU) = LL(VU|W) + LL(W)
    model.logLik=LogLikVUgW+LogLikW;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L chCov]=doLogLik(Sigma,data)
  chCov=chol(Sigma);
  logDet=sum(log(diag(chCov))); % actually, the log sqrt(det)
  p1=(chCov')\data;
  L=-logDet-0.5*(p1'*p1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L V Dki2]=doLogLikSep(Sigma,nugget,data)

  % get the eigenDecomp for the blocks
    for ii=1:length(Sigma)
      [V{ii} D{ii}]=eig(Sigma{ii});
    end
  
  % compute the determinant from the eigenvalues and the nugget
    dkron=diag(D{end});
    for ii=(length(D)-1):-1:1
      dkron=kron(diag(D{ii}),dkron);
    end
    logDet=sum(log(dkron+nugget));

  % compute the composite inverse, including the nugget
    %vkron=V{end};
    %for ii=(length(V)-1):-1:1
    %  vkron=kron(V{ii},vkron);
    %end
    %Sinv=vkron * diag(1./(dkron+nugget)) * vkron';

  % alternate implementation to indirectly compute data'*Sinv*data
    zp=sepQuadFormCalc(V,data);
    Dki2=1./sqrt(dkron + nugget);
    zp2=zp .* Dki2;
    
  % compute the log likelihood
    %L=-0.5*logDet-0.5*data'*Sinv*data;   % make this more efficient someday
    L=-0.5*logDet-0.5*zp2'*zp2;   % here it is ...

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zp=sepQuadFormCalc(V,zp)
  % calculate right side of the kronecker quadratic form solve
    del=numel(zp);
    for ii=length(V):-1:1
      Vsize=size(V{ii},2);
      zp=( V{ii}\reshape(zp,[Vsize del/Vsize]) )';
    end
  % final reshape of zp to the total matrix size
  Vsize=1; for ii=1:numel(V); Vsize=Vsize*size(V{ii},1); end
  zp=reshape(zp,[Vsize del/Vsize]);
end

