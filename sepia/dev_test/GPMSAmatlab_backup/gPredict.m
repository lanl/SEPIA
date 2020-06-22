%function pred=gPredict(xpred,pvals,model,data,varargs)
% Predict using a gpmsa constructed model. 
% result is a 3-dimensional prediction matrix:
%       #pvals by model-dims by length-xpred 
%   model-dims is the simulation basis size for a w-prediction (a model
%   with no observation data) or the v (discrepancy basis loadings) and u 
%   (simulation basis loadings) for uv-prediction
% argument pair keywords are
%   mode - 'wpred' or 'uvpred'
%   theta - the calibration variable value(s) to use, either one value or
%           one for each xpred. Default is the calibrated param in pvals
%   addResidVar - default 0, whether to add the residual variability
%   returnRealization - default 1, whether to return realizations of the
%           process specified. values will be in a .w field for wpred, or
%           .v and .u fields for uvpred.
%   returnMuSigma - default 0, whether to return the mean and covariance of
%           the process specified. results will be in a matrix .Myhat field
%           and a cell array .Syhat field. 
%   returnCC - default 0, return the cross-covariance of the data and
%           predictors.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pred=gPredict(xpred,pvals,model,data,varargin)

  if model.n==0
    mode='wpred';
  else
    mode='uvpred';
  end
  theta=[];
  addResidVar=0;
  returnRealization=1;
  returnMuSigma=0;
  returnCC=0;
  parseAssignVarargs({'mode','theta','addResidVar', ...
                      'returnRealization','returnMuSigma', ...
                      'returnCC'}); 

  % backwards compat for categorical capability
  if ~isfield(data,'catInd'); data.catInd=[]; end

  switch mode
    case 'uvpred'
        pred=uvpredSepAware(xpred,pvals,model,data,theta,addResidVar, ...
                    returnRealization,returnMuSigma);
    case 'wpred'
        if ~isempty(data.ztSep) 
          error('Separable design predictions are not implemented for wpred');
        end
        pred=wPred(xpred,pvals,model,data,theta,addResidVar, ...
                    returnRealization,returnMuSigma,returnCC);
    case 'uvpredNoblock'
        pred=uvPred(xpred,pvals,model,data,theta,addResidVar, ...
                    returnRealization,returnMuSigma,returnCC);
    otherwise
      error('invalid mode in gPredict');
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pred=wPred(xpred,pvals,model,data,thetapred,addResidVar,retRlz,retMS,retC)

  n=model.n; m=model.m;p=model.p;q=model.q;pu=model.pu;

  debugRands=false;
  if isfield(model,'debugRands'); debugRands=model.debugRands; end
  
  npred=size(xpred,1);

  diags1=diagInds(m*pu);
  diags2=diagInds(npred*pu);

  nreal=length(pvals);
  tpred=zeros([nreal,npred*pu]);

  for ii=1:length(pvals)
    if n>0
      theta=pvals(ii).theta'; 
    end
    betaU=reshape(pvals(ii).betaU,p+q,pu);
    lamUz=pvals(ii).lamUz;
    lamWs=pvals(ii).lamWs; lamWOs=pvals(ii).lamWOs;

    if n>0
      if isempty(thetapred)
        xpredt=[xpred repmat(theta,npred,1)];
      else
        xpredt=[xpred thetapred];
      end
    else
      xpredt=xpred;
    end

    xpredDist=genDist(xpredt,data.catInd);  
    zxpredDist=genDist2(data.zt,xpredt,data.catInd);

      SigW=zeros(m*pu);
        for jj=1:pu
          bStart=(jj-1)*m+1; bEnd=bStart+m-1; 
          SigW(bStart:bEnd,bStart:bEnd)=...
              gCovMat(model.ztDist,betaU(:,jj),lamUz(jj));
        end
        SigW(diags1)=SigW(diags1)+ ...
            kron(1./(model.LamSim*lamWOs)',ones(1,m)) + ...
            kron(1./(lamWs)',ones(1,m)) ;

      SigWp=zeros(npred*pu);
        for jj=1:pu
          bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
          SigWp(bStart:bEnd,bStart:bEnd)= ...
              gCovMat(xpredDist,betaU(:,jj),lamUz(jj));
        end
        SigWp(diags2)=SigWp(diags2)+ ...
             kron(1./(lamWs)',ones(1,npred)) ;
        if addResidVar
          SigWp(diags2)=SigWp(diags2)+ ...
             kron(1./(model.LamSim*lamWOs)',ones(1,npred));
        end
      SigWWp=zeros(m*pu,npred*pu);
        for jj=1:pu
          bStartI=(jj-1)*m+1; bEndI=bStartI+m-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigWWp(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(zxpredDist,betaU(:,jj),lamUz(jj));
        end

    SigData=SigW;
    SigPred=SigWp;
    SigCross=SigWWp;
    % Get the stats for the prediction stuff. 
      %W=(SigCross')/SigData;
      W=linsolve(SigData,SigCross,struct('SYM',true,'POSDEF',true))';
      Myhat=W*(data.w(:));
      Syhat=SigPred-W*SigCross;
      
    if retRlz
      % And do a realization
      tpred(ii,:)=rmultnormsvd(1,Myhat,Syhat',debugRands)';
    end
    if retMS
      % add the distribution params
      pred.Myhat(ii,:)=Myhat;
      pred.Syhat{ii}=Syhat;
    end
    if retC
      pred.CC{ii}=SigCross;
    end
  end
  
  if retRlz
    % Reshape the pred matrix to 3D:
    %  first dim  - (number of realizations [pvals])
    %  second dim - (number of principal components)
    %  third dim  - (number of points [x,theta]s) 
    pred.w=zeros(length(pvals),pu,npred);
    for ii=1:pu
      pred.w(:,ii,:)=tpred(:,(ii-1)*npred+1:ii*npred);
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pred=uvPred(xpred,pvals,model,data,thetapred,addResidVar,retRlz,retMS,retC)

  n=model.n;m=model.m;p=model.p;q=model.q;pu=model.pu;pv=model.pv;
  lamVzGnum=model.lamVzGnum; lamVzGroup=model.lamVzGroup;

  debugRands=false;
  if isfield(model,'debugRands'); debugRands=model.debugRands; end

  npred=size(xpred,1);

  diags0=diagInds(n*pu);
  diags1=diagInds(m*pu);
  diags2=diagInds(npred*pu);

  x0Dist=genDist(data.x,data.catInd);
  xpred0Dist=genDist(xpred,data.catInd);
  xxpred0Dist=genDist2(data.x,xpred,data.catInd);

  nreal=length(pvals);
  tpred=zeros([nreal,npred*(pv+pu)]);

  vCov(lamVzGnum).mat=[];
  vpCov(lamVzGnum).mat=[];
  vvCov(lamVzGnum).mat=[];

  for ii=1:length(pvals)
    theta=pvals(ii).theta';
    betaV=reshape(pvals(ii).betaV,p,lamVzGnum); 
    betaU=reshape(pvals(ii).betaU,p+q,pu);
    lamVz=pvals(ii).lamVz; lamUz=pvals(ii).lamUz; lamWOs=pvals(ii).lamWOs;
    lamWs=pvals(ii).lamWs; lamOs=pvals(ii).lamOs;

    if isempty(thetapred)
      xpredt=[xpred repmat(theta,npred,1)];
    else
      xpredt=[xpred thetapred];
    end

    xDist=genDist([data.x repmat(theta,n,1)],data.catInd);
    ztDist=genDist(data.zt,data.catInd);
    xzDist=genDist2([data.x repmat(theta,n,1)],data.zt,data.catInd);
    xpredDist=genDist(xpredt,data.catInd);  
    xxpredDist=genDist2([data.x repmat(theta,n,1)],xpredt,data.catInd);
    zxpredDist=genDist2(data.zt,xpredt,data.catInd);

    % Generate the part of the matrix related to the data
    % Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
      SigV=zeros(n*pv);
        for jj=1:lamVzGnum;
          vCov(jj).mat=gCovMat(x0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStart=(jj-1)*n+1; bEnd=bStart+n-1;
          SigV(bStart:bEnd,bStart:bEnd)=vCov(lamVzGroup(jj)).mat;
        end
      SigU=zeros(n*pu);
        for jj=1:pu
          bStart=(jj-1)*n+1; bEnd=bStart+n-1;
          SigU(bStart:bEnd,bStart:bEnd)= ...
              gCovMat(xDist,betaU(:,jj),lamUz(jj));
        end
        SigU(diags0)=SigU(diags0)+...
             kron(1./(lamWs)',ones(1,n)) ;
      SigW=zeros(m*pu);
        for jj=1:pu
          bStart=(jj-1)*m+1; bEnd=bStart+m-1; 
          SigW(bStart:bEnd,bStart:bEnd)=...
              gCovMat(ztDist,betaU(:,jj),lamUz(jj));
        end
        SigW(diags1)=SigW(diags1)+ ...
            kron(1./(model.LamSim*lamWOs)',ones(1,m)) + ...
            kron(1./(lamWs)',ones(1,m)) ;
      SigUW=zeros(n*pu,m*pu);
        for jj=1:pu
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*m+1; bEndJ=bStartJ+m-1;
          SigUW(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(xzDist,betaU(:,jj),lamUz(jj));
        end
        if model.scOut
          SigData=[ SigU+SigV    SigUW; ...
                    SigUW'       SigW ];
          SigData(1:n*pu,1:n*pu) = ...
            SigData(1:n*pu,1:n*pu) + model.SigObs*1/lamOs;
        else
          SigData=[SigV                 zeros(n*pv,(n+m)*pu);  ...
                   zeros((n+m)*pu,n*pv) [ SigU    SigUW; ...
                                          SigUW'  SigW  ] ];
          SigData(1:n*(pv+pu),1:n*(pv+pu)) = ...
            SigData(1:n*(pv+pu),1:n*(pv+pu)) + model.SigObs*1/lamOs;
        end

    % Generate the part of the matrix related to the predictors
    % Parts to compute: Sig_vpred, Sig_upred
      SigVp=zeros(npred*pv);
        for jj=1:lamVzGnum
          vpCov(jj).mat=gCovMat(xpred0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
          SigVp(bStart:bEnd,bStart:bEnd)=vpCov(lamVzGroup(jj)).mat;
        end
        %SigVp(diagInds(npred*pv))=SigVp(diagInds(npred*pv))+1;
      SigUp=zeros(npred*pu);
        for jj=1:pu
          bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
          SigUp(bStart:bEnd,bStart:bEnd)= ...
              gCovMat(xpredDist,betaU(:,jj),lamUz(jj));
        end
        SigUp(diags2)=SigUp(diags2)+...
             kron(1./(lamWs)',ones(1,npred)) ;
        if addResidVar
          SigUp(diags2)=SigUp(diags2)+ ...
             kron(1./(model.LamSim*lamWOs)',ones(1,npred)) ;
        end
           
      SigPred=[SigVp                     zeros(npred*pv,npred*pu);  ...
               zeros(npred*pu,npred*pv)  SigUp  ];

    % Now the cross-terms. 
      SigVVx=zeros(n*pv,npred*pv);    
        for jj=1:lamVzGnum;
          vvCov(jj).mat=gCovMat(xxpred0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigVVx(bStartI:bEndI,bStartJ:bEndJ)=vvCov(lamVzGroup(jj)).mat; 
        end
      SigUUx=zeros(n*pu,npred*pu);
        for jj=1:pu
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigUUx(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(xxpredDist,betaU(:,jj),lamUz(jj));
        end
      SigWUx=zeros(m*pu,npred*pu);
        for jj=1:pu
          bStartI=(jj-1)*m+1; bEndI=bStartI+m-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigWUx(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(zxpredDist,betaU(:,jj),lamUz(jj));
        end
      if model.scOut
        SigCross=[SigVVx                 SigUUx; ...
                  zeros(m*pu,npred*pv)   SigWUx];
      else
        SigCross=[SigVVx                 zeros(n*pv,npred*pu); ...
                  zeros(n*pu,npred*pv)   SigUUx; ...
                  zeros(m*pu,npred*pv)   SigWUx];
      end

    % Get the stats for the prediction stuff. 
      %W=(SigCross')/SigData;
      W=linsolve(SigData,SigCross,struct('SYM',true,'POSDEF',true))';
      if model.scOut, Myhat=W*model.uw; else Myhat=W*model.vuw; end
      Syhat=SigPred-W*SigCross;
      
    if retRlz
      % And do a realization
      tpred(ii,:)=rmultnormsvd(1,Myhat,Syhat',debugRands)';
    end
    if retMS
      % log the distribution params
      pred.Myhat(ii,:)=Myhat;
      pred.Syhat{ii}=Syhat;
    end
    if retC
      pred.CC{ii}=SigCross;
    end
  end

  if retRlz
    % Reshape the pred matrix to 3D, for each component:
    %  first dim  - (number of realizations [pvals])
    %  second dim - (number of principal components)
    %  third dim  - (number of points [x,theta]s) 
    pred.v=zeros(length(pvals),pv,npred);
    pred.u=zeros(length(pvals),pu,npred);
    for ii=1:pv
      pred.v(:,ii,:)=tpred(:,(ii-1)*npred+1:ii*npred);
    end
    for ii=1:pu
      pred.u(:,ii,:)=tpred(:,pv*npred+((ii-1)*npred+1:ii*npred) );
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pred=uvpredSepAware(xpred,pvals,model,data,thetapred,addResidVar,retRlz,retMS)

  n=model.n;m=model.m;p=model.p;q=model.q;pu=model.pu;pv=model.pv;
  lamVzGnum=model.lamVzGnum; lamVzGroup=model.lamVzGroup;

  debugRands=false;
  if isfield(model,'debugRands'); debugRands=model.debugRands; end

  npred=size(xpred,1);

  diags0=diagInds(n*pu);
  diags2=diagInds(npred*pu);

  diagsm=diagInds(m);
  
  x0Dist=genDist(data.x,data.catInd);
  xpred0Dist=genDist(xpred,data.catInd);
  xxpred0Dist=genDist2(data.x,xpred,data.catInd);

  nreal=length(pvals);
  tpred=zeros([nreal,npred*(pv+pu)]);

  vCov(lamVzGnum).mat=[];
  vpCov(lamVzGnum).mat=[];
  vvCov(lamVzGnum).mat=[];
  SigWsb(pu).mat=[];
  mugWsb(pu).vec=[];
  SigWUxb(pu).mat=[];
  SigWb(pu).kmats=[]; SigWb(pu).kDiag=[];
  SigUWb(pu).mat=[];

  for ii=1:length(pvals)
    theta=pvals(ii).theta';
    betaV=reshape(pvals(ii).betaV,p,lamVzGnum); 
    betaU=reshape(pvals(ii).betaU,p+q,pu);
    lamVz=pvals(ii).lamVz; lamUz=pvals(ii).lamUz; lamWOs=pvals(ii).lamWOs;
    lamWs=pvals(ii).lamWs; lamOs=pvals(ii).lamOs;

    if isempty(thetapred)
      xpredt=[xpred repmat(theta,npred,1)];
    else
      xpredt=[xpred thetapred];
    end

    xDist=genDist([data.x repmat(theta,n,1)],data.catInd);
    ztDist=genDist(data.zt,data.catInd);
    xzDist=genDist2([data.x repmat(theta,n,1)],data.zt,data.catInd);
    xpredDist=genDist(xpredt,data.catInd);  
    xxpredDist=genDist2([data.x repmat(theta,n,1)],xpredt,data.catInd);
    zxpredDist=genDist2(data.zt,xpredt,data.catInd);

    % Generate the part of the matrix related to the data
    % Four parts to compute: Sig_v, Sig_u, Sig_w, and the Sig_uw crossterm
      SigV=zeros(n*pv);
        for jj=1:lamVzGnum;
          vCov(jj).mat=gCovMat(x0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStart=(jj-1)*n+1; bEnd=bStart+n-1;
          SigV(bStart:bEnd,bStart:bEnd)=vCov(lamVzGroup(jj)).mat;
        end
      SigU=zeros(n*pu);
        for jj=1:pu
          bStart=(jj-1)*n+1; bEnd=bStart+n-1;
          SigU(bStart:bEnd,bStart:bEnd)= ...
              gCovMat(xDist,betaU(:,jj),lamUz(jj));
        end
        SigU(diags0)=SigU(diags0)+...
             kron(1./(lamWs)',ones(1,n)) ;
      SigUW=zeros(n*pu,m*pu);
        for jj=1:pu
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*m+1; bEndJ=bStartJ+m-1;
          SigUW(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(xzDist,betaU(:,jj),lamUz(jj));
        end

      if isempty(data.ztSep)
          for jj=1:pu
            SigWb(jj).mat=gCovMat(ztDist,betaU(:,jj),lamUz(jj));
            SigWb(jj).mat(diagsm)=SigWb(jj).mat(diagsm)+...
                1/(model.LamSim(jj)*lamWOs) + 1/lamWs(jj);  %scalar nugget for each block
          end
      else
          for jj=1:pu
            segVarStart=1;
            for kk=1:length(data.ztSep)
              segVars=segVarStart:(segVarStart + model.ztSepDist{kk}.p-1);
              segVarStart=segVarStart+ model.ztSepDist{kk}.p;
              if (kk==1)  % ensure that lamUz is only counted once
                SigWb(jj).kmat{kk}=gCovMat(model.ztSepDist{kk},betaU(segVars,jj),lamUz(jj));
              else
                SigWb(jj).kmat{kk}=gCovMat(model.ztSepDist{kk},betaU(segVars,jj),1);
              end
            end
            SigWb(jj).kdiag=1/(model.LamSim(jj)*lamWOs) + 1/lamWs(jj);
          end
      end
      for jj=1:pu
        SigUWb(jj).mat=[zeros((jj-1)*n,m);
                        gCovMat(xzDist,betaU(:,jj),lamUz(jj));
                        zeros((pu-jj)*n,m) ];
      end
      
      %ksep Scalar output modeling is not implemented ... used to be this:
        %if model.scOut
        %  SigData=[ SigU+SigV    SigUW; ...
        %            SigUW'       SigW ];
        %  SigData(1:n*pu,1:n*pu) = ...
        %    SigData(1:n*pu,1:n*pu) + model.SigObs*1/lamOs;
        %else
        %  SigData=[SigV                 zeros(n*pv,(n+m)*pu);  ...
        %           zeros((n+m)*pu,n*pv) [ SigU    SigUW; ...
        %                                  SigUW'  SigW  ] ];
        %  SigData(1:n*(pv+pu),1:n*(pv+pu)) = ...
        %    SigData(1:n*(pv+pu),1:n*(pv+pu)) + model.SigObs*1/lamOs;
        %end
        %ksep

    % Generate the part of the matrix related to the predictors
    % Parts to compute: Sig_vpred, Sig_upred
      SigVp=zeros(npred*pv);
        for jj=1:lamVzGnum;
          vpCov(jj).mat=gCovMat(xpred0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
          SigVp(bStart:bEnd,bStart:bEnd)=vpCov(lamVzGroup(jj)).mat;
        end
      SigUp=zeros(npred*pu);
        for jj=1:pu
          bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
          SigUp(bStart:bEnd,bStart:bEnd)= ...
              gCovMat(xpredDist,betaU(:,jj),lamUz(jj));
        end
        SigUp(diags2)=SigUp(diags2)+...
             kron(1./(lamWs)',ones(1,npred)) ;
        if addResidVar
          SigUp(diags2)=SigUp(diags2)+ ...
             kron(1./(model.LamSim*lamWOs)',ones(1,npred)) ;
        end
           
      SigPred=[SigVp                     zeros(npred*pv,npred*pu);  ...
               zeros(npred*pu,npred*pv)  SigUp  ];

    % Now the cross-terms. 
      SigVVx=zeros(n*pv,npred*pv);    
        for jj=1:lamVzGnum
          vvCov(jj).mat=gCovMat(xxpred0Dist, betaV(:,jj), lamVz(jj));
        end
        for jj=1:pv
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigVVx(bStartI:bEndI,bStartJ:bEndJ)=vvCov(lamVzGroup(jj)).mat; 
        end
      SigUUx=zeros(n*pu,npred*pu);
        for jj=1:pu
          bStartI=(jj-1)*n+1; bEndI=bStartI+n-1;
          bStartJ=(jj-1)*npred+1; bEndJ=bStartJ+npred-1;
          SigUUx(bStartI:bEndI,bStartJ:bEndJ)=...
              gCovMat(xxpredDist,betaU(:,jj),lamUz(jj));
        end
        
        for jj=1:pu
          SigWUxb(jj).mat=[zeros((jj-1)*npred,m) ;
                           gCovMat(zxpredDist,betaU(:,jj),lamUz(jj))';
                           zeros((pu-jj)*npred,m) ];
        end
      
      % ksep
      %if model.scOut
      %  SigCross=[SigVVx                 SigUUx; ...
      %            zeros(m*pu,npred*pv)   SigWUx];
      %else
      %  SigCross=[SigVVx                 zeros(n*pv,npred*pu); ...
      %            zeros(n*pu,npred*pv)   SigUUx; ...
      %            zeros(m*pu,npred*pv)   SigWUx];
      %end
      % ksep

    % Get the stats for the prediction stuff. 
      %W=(SigCross')/SigData;
      % ksep
      %W=linsolve(SigData,SigCross,struct('SYM',true,'POSDEF',true))';
      %if model.scOut, Myhat=W*model.uw; else Myhat=W*model.vuw; end
      %Syhat=SigPred-W*SigCross;
      % ksep

      SigVUo=[SigV             zeros(n*pv,n*pu);  ...
              zeros(n*pu,n*pv) SigU ] ...
             + model.SigObs*1/lamOs;
      SigVUx=[ SigVVx                            zeros(n*pv,npred*pu); ...
               zeros(n*pu,npred*pv)  SigUUx];
      SignoW=[SigVUo  SigVUx; ...
              SigVUx' SigPred];

      for jj=1:pu
        SigWcrossb=[zeros(n*pv,m); 
                   SigUWb(jj).mat; 
                   zeros(npred*pv,m);
                   SigWUxb(jj).mat]';
        if isempty(data.ztSep)
          Tb=linsolve(SigWb(jj).mat,SigWcrossb,struct('SYM',true,'POSDEF',true))';
          SigWsb(jj).mat= Tb * SigWcrossb;
          mugWsb(jj).vec= Tb * model.w((1:m)+(jj-1)*m);
        else
          [SigWsb(jj).mat, mugWsb(jj).vec]= ...
             sepCalc(SigWcrossb,SigWb(jj).kmat,SigWb(jj).kdiag,model.w((1:m)+(jj-1)*m) );
        end
      end
      
      SiggWb=SignoW;
      mugWb=zeros(size(mugWsb(1).vec));
      for jj=1:pu
        SiggWb=SiggWb - SigWsb(jj).mat;
        mugWb=mugWb + mugWsb(jj).vec;
      end

      SiggW11=SiggWb(1:n*(pv+pu),1:n*(pv+pu));
      SiggW22=SiggWb(n*(pv+pu)+1:end,n*(pv+pu)+1:end);
      SiggW12=SiggWb(1:n*(pv+pu),n*(pv+pu)+1:end);
      mugW1=mugWb(1:n*(pv+pu));
      mugW2=mugWb(n*(pv+pu)+1:end);

      T=linsolve(SiggW11,SiggW12,struct('SYM',true,'POSDEF',true))';
      Syhat=SiggW22 - T*SiggW12;
      Myhat=mugW2 + T*(model.vu - mugW1);

    if retRlz
      % draw a realization
      tpred(ii,:)=rmultnormsvd(1,Myhat,Syhat',debugRands)';
    end
    if retMS
      % log the distribution params
      pred.Myhat(ii,:)=Myhat;
      pred.Syhat{ii}=Syhat;
    end
  end

  if retRlz
    % Reshape the pred matrix to 3D, for each component:
    %  first dim  - (number of realizations [pvals])
    %  second dim - (number of principal components)
    %  third dim  - (number of points [x,theta]s) 
    pred.v=zeros(length(pvals),pv,npred);
    pred.u=zeros(length(pvals),pu,npred);
    for ii=1:pv
      pred.v(:,ii,:)=tpred(:,(ii-1)*npred+1:ii*npred);
    end
    for ii=1:pu
      pred.u(:,ii,:)=tpred(:,pv*npred+((ii-1)*npred+1:ii*npred) );
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [T, r]=sepCalc(W,S,nugget,m)
  % use kron form to compute: 
  %  T = W' * inv( kron(S) + nugget*I ) * W
  %  r = W' * inv( kron(S) + nugget*I ) * m
  % for matrix W, covariance S, scalar nugget, and vector m
  % where S is a cell array of sub-covariances to be composed by kron 

    % get the eigenDecomp for the blocks
    V=cell(length(S),1); D=cell(length(S),1);
    for ii=1:length(S)
      [V{ii}, D{ii}]=eig(S{ii});
    end

    % get the composite diagonal from the eigenvalues
    dkron=diag(D{end});
    for ii=(length(D)-1):-1:1
      dkron=kron(diag(D{ii}),dkron);
    end

    % inverse sqrt of the D diagonal augmented with the nugget
    Dki2=1./sqrt(dkron + nugget);
    
    % Put the parts together for W'inv(S)W
    zp=sepQuadFormCalc(V,W);
    zp2T=zp .* repmat(Dki2,[1 size(zp,2)] );
    T=zp2T'*zp2T;

    % Put the parts together for W'inv(S)m
    zp=sepQuadFormCalc(V,m);
    zp2r=zp .* Dki2;
    r=zp2T'*zp2r;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=sepQuadFormCalc(V,W)
  % calculate right side of the kronecker quadratic form solve
  osize=1; for ii=1:length(V); osize=osize*size(V{ii},1); end
  p=zeros(osize,size(W,2));
  for jj=1:size(W,2) % need to do this for each column of W
    zp=W(:,jj);
    zel=numel(zp);
    for ii=length(V):-1:1
      Vsize=size(V{ii},2);
      zp=( V{ii}\reshape(zp,[Vsize zel/Vsize]) )';
    end
    p(:,jj)=zp(:); % the final reshape of zp is to a column vector
  end
end


%
% Helper function rmultnormSVD computes multivariate normal realizations
function rnorm = rmultnormsvd(n,mu,cov,debugRands)
    %rnorm=mvnrnd(mu,cov);

  if exist('debugRands','var') && debugRands
    % for comparison (e.g. with python), recreatable mvns
    normalrands=norminv(rand(size(mu,1),n));
  else
    normalrands=randn(size(mu,1),n);
  end
    
  [U, S] = svd(cov);
  rnorm = repmat(mu,1,n) + U*sqrt(S) * normalrands;

end 
