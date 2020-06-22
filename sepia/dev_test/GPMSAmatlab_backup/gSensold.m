% function sens=gSens(pout,varargin)
% Compute sensitivity functions and indices
%  pout - MCMC output object
%  varargs are in string/value pairs
%    'pvec' ==> subsample of MCMC iterates used for sensitivity calculations
%               (default - all iterates)
%    'ngrid' - default 21 ==> number of grid points in each dimension for
%                             calculation of main/joint effect functions
%    'varlist' ==> matrix with pairs of variables in each row for which
%                  joint effects are desired (default - empty)
%                  'all' - compute joint effects for all pairs of variables
%    'jelist' ==> cell array with row vectors indicating variables for
%                 which joint sensitivities are desired (default - empty)
%    'rg' ==> matrix with one row for each variable, giving min/max values
%             for sensitivity calculations assuming unit hypercube scaling
%             (default - unit hypercube)
%    'option' ==> sensitivity calculations based on ...
%                 'mean' - posterior mean GP parameters
%                 'median' - posterior median GP parameters
%                 params - user provided GP parameters; must have betaU,
%                          lamUz and lamWs fields (row vectors)
%                 (default - empty)
%
% For functional/multivariate analyses, sensitivity indicies are correct
% assuming an orthogonal basis, i.e. pout.simData.Ksim is an orthogonal
% matrix.
%
% For multivariate analyses, i.e. collection of features with possibly
% different variances, sensitivity indices are computed relative to the
% output scaling assumed to generate the basis matrix (pout.simData.Ksim).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
%         Brian Williams, Los Alamos National Laboratory
%
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079
%
% © Copyright Los Alamos National Security, LLC.
%
% This Software was produced under a U.S. Government contract 
% (DE-AC52-06NA25396) by Los Alamos National Laboratory, which is operated 
% by the Los Alamos National Security, LLC (LANS) for the U.S. Department 
% of Energy, National Nuclear Security Administration. The U.S. Government
% is licensed to use, reproduce, and distribute this Software. Permission
% is granted to the public to copy and use this Software without charge,
% provided that this Notice and any statement of authorship are reproduced 
% on all copies. Neither the Government nor the LANS makes any warranty, 
% express or implied, or assumes any liability or responsibility for the 
% user of this Software. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sens=gSens(pout,varargin)
% sensitivity analysis

% process input arguments
 pvec=1:length(pout.pvals); ngrid=21; varlist=[]; jelist=[]; rg=[]; option=[];
 parseAssignVarargs({'pvec','ngrid','varlist','jelist','rg','option'});

 p=pout.model.p; q=pout.model.q; nv=p+q; pu=pout.model.pu;

 ii0=[];
 if isempty(rg)
   rg=zeros(nv,2); rg(:,2)=1;
 else
   ii0=find(rg(:,1) == rg(:,2));
   if ~isempty(ii0)
     ii0=setxor(1:nv,ii0); nv=length(ii0); rg=rg(ii0,:);
   end
 end
 if (nv==0), fprintf('error:  no free variables\n'); return; end
 if isempty(ii0), ii0=1:nv; end

 xe=zeros(ngrid,nv);
 for ii=1:nv
    xe(:,ii)=[rg(ii,1):(rg(ii,2)-rg(ii,1))/(ngrid-1):rg(ii,2)]';
 end
 
 %kriging
 BetaU=[pout.pvals(pvec).betaU]';
 LamUz=[pout.pvals(pvec).lamUz]'; LamWs=[pout.pvals(pvec).lamWs]';
 if ~isempty(option)
   if ischar(option)
     if ~strcmp(option,'mean') & ~strcmp(option,'median')
       fprintf('error:  invalid option\n'); return;
     end
     BetaU=feval(option,BetaU);
     LamUz=feval(option,LamUz); LamWs=feval(option,LamWs);
   else
     if ~all(isfield(option,{'betaU','lamUz','lamWs'}))
       fprintf('error:  option missing required fields\n'); return;
     end
     BetaU=option.betaU; LamUz=option.lamUz; LamWs=option.lamWs;
   end
   npvec=size(BetaU,1);
 else
    npvec=length(pvec);
 end
 
 if ~isempty(varlist)
    if ischar(varlist), if strcmp(varlist,'all'),
       kk=0; varlist=[];
       for ii=1:nv-1, for jj=ii+1:nv, kk=kk+1; varlist(kk,:)=[ii jj]; end; end
    end; end
 end

 for ii=1:pu
    betaU=BetaU(:,ii0+(ii-1)*nv); lamUz=LamUz(:,ii); lamWs=LamWs(:,ii);
    sa(ii)=componentSens(pout.simData.x(:,ii0),pout.data.w(:,ii),betaU,...
                         lamUz,lamWs,xe,ngrid,varlist,jelist,rg,pout.data);
 end

 ymean=pout.simData.orig.ymean; ysd=pout.simData.orig.ysd;
 if length(ysd)==1, ysd=repmat(ysd,[size(ymean,1) 1]); end

 ksmm=pout.simData.Ksim;
 lam=diag(ksmm'*ksmm);
 sme=zeros(npvec,nv);
 ste=zeros(npvec,nv);
 for ii=1:npvec
    vt0=0;
    for jj=1:pu
       sme(ii,:)=sme(ii,:)+lam(jj)*sa(jj).sme(ii,:)*sa(jj).vt(ii);
       ste(ii,:)=ste(ii,:)+lam(jj)*sa(jj).ste(ii,:)*sa(jj).vt(ii);
       vt0=vt0+lam(jj)*sa(jj).vt(ii);
    end
    sme(ii,:)=sme(ii,:)/vt0;
    ste(ii,:)=ste(ii,:)/vt0;
    vt(ii)=vt0;
 end
 smePm=squeeze(mean(sme,1));
 stePm=squeeze(mean(ste,1));

 if ~isempty(varlist)
    sie=zeros(npvec,size(varlist,1));
    for ii=1:npvec
       for jj=1:pu
          sie(ii,:)=sie(ii,:)+lam(jj)*sa(jj).sie(ii,:)*sa(jj).vt(ii);
       end
       sie(ii,:)=sie(ii,:)/vt(ii);
    end
    siePm=squeeze(mean(sie,1));
 end
 
 if ~isempty(jelist)
    sje=zeros(npvec,length(jelist));
    for ii=1:npvec
       for jj=1:pu
          sje(ii,:)=sje(ii,:)+lam(jj)*sa(jj).sje(ii,:)*sa(jj).vt(ii);
       end
       sje(ii,:)=sje(ii,:)/vt(ii);
    end
    sjePm=squeeze(mean(sje,1));
 end

%unscaled
 e0=zeros(size(ksmm,1),1);
 mef.m=zeros([pu nv size(ksmm,1) ngrid]);
 mef.sd=zeros([pu nv size(ksmm,1) ngrid]);

 meanmat=repmat(ymean,[1 ngrid]); ysdmat=repmat(ysd,[1 ngrid]);

 for jj=1:pu
    e0=e0+ksmm(:,jj).*mean(sa(jj).e0);
    for kk=1:nv
       mef.m(jj,kk,:,:)=kron(ksmm(:,jj),...
                             reshape(mean(sa(jj).mef.m(:,kk,:),1),[],...
                                     ngrid)).*ysdmat+meanmat;
       mef.sd(jj,kk,:,:)=sqrt(kron(ksmm(:,jj).^2,...
                              reshape(var(sa(jj).mef.m(:,kk,:),0,1),[],...
                                      ngrid))+kron(ksmm(:,jj).^2,...
                              reshape(mean(sa(jj).mef.v(:,kk,:),1),[],...
                                      ngrid))).*ysdmat;
    end
 end
 e0=e0.*ysd+ymean;

 a=size(mef.m); tmef.m=reshape(sum(mef.m,1),a([2 3 4]));
 for kk=1:nv
    tmef.m(kk,:,:)=reshape(tmef.m(kk,:,:),a([3 4]))-(pu-1)*meanmat;
 end
 tmef.m=squeeze(tmef.m);
 tmef.sd=zeros(a([2 3 4]));
 for jj=1:nv
    for kk=1:pu
       tmef.sd(jj,:,:)=reshape(tmef.sd(jj,:,:),[],ngrid)+...
                       kron(ksmm(:,kk).^2,...
                       reshape(mean(sa(kk).mef.v(:,jj,:),1),[],ngrid));
    end
 end
 tmp=zeros([npvec a([2 4])]);
 for ii=1:size(ksmm,1)
    for jj=1:nv
       for kk=1:pu
          tmp(:,jj,:)=reshape(tmp(:,jj,:),[],ngrid)+ksmm(ii,kk)*...
                      reshape(sa(kk).mef.m(:,jj,:),[],ngrid);
       end
       tmef.sd(jj,ii,:)=tmef.sd(jj,ii,:)+var(tmp(:,jj,:),0,1);
    end
 end
 for kk=1:nv
    tmef.sd(kk,:,:)=sqrt(reshape(tmef.sd(kk,:,:),a([3 4]))).*ysdmat;
 end
 tmef.sd=squeeze(tmef.sd);

 if ~isempty(varlist)
    jef.m=zeros([pu size(varlist,1) ngrid*size(ksmm,1) ngrid]);
    jef.sd=zeros([pu size(varlist,1) ngrid*size(ksmm,1) ngrid]);
    meanmat=repmat(ymean,[ngrid ngrid]); ysdmat=repmat(ysd,[ngrid ngrid]);
    for jj=1:pu
       for kk=1:size(varlist,1)
          jef.m(jj,kk,:,:)=kron(reshape(mean(sa(jj).jef.m(:,kk,:,:),1),...
                                        [],ngrid),ksmm(:,jj)).*ysdmat+ ...
                           meanmat;
          jef.sd(jj,kk,:,:)=sqrt(kron(reshape(var(sa(jj).jef.m(:,kk,:,:),...
                                              0,1),[],ngrid),...
                                 ksmm(:,jj).^2)+...
                                 kron(reshape(mean(sa(jj).jef.v(:,kk,:,:),...
                                              1),[],ngrid),...
                                 ksmm(:,jj).^2)).*ysdmat;
      end
    end

    a=size(jef.m); tjef.m=reshape(sum(jef.m,1),a([2 3 4]));
    for kk=1:size(varlist,1)
       tjef.m(kk,:,:)=reshape(tjef.m(kk,:,:),a([3 4]))-(pu-1)*meanmat;
    end
    tjef.m=squeeze(tjef.m);
    tjef.sd=zeros(a([2 3 4]));
    for jj=1:size(varlist,1)
       for kk=1:pu
          tjef.sd(jj,:,:)=reshape(tjef.sd(jj,:,:),[],ngrid)+...
                          kron(reshape(mean(sa(kk).jef.v(:,jj,:,:),1),...
                                       [],ngrid),ksmm(:,kk).^2);
       end
    end
    tmp=zeros([npvec a([2 4])]);
    for hh=1:ngrid
       for ii=1:size(ksmm,1)
          for jj=1:size(varlist,1)
             for kk=1:pu
                tmp(:,jj,:)=reshape(tmp(:,jj,:),[],ngrid)+ksmm(ii,kk)*...
                            reshape(sa(kk).jef.m(:,jj,hh,:),[],ngrid);
             end
             tjef.sd(jj,(hh-1)*size(ksmm,1)+ii,:)=tjef.sd(jj,(hh-1)*...
                                                  size(ksmm,1)+ii,:)+...
                                                  var(tmp(:,jj,:),0,1);
          end
       end
    end
    for kk=1:size(varlist,1)
       tjef.sd(kk,:,:)=sqrt(reshape(tjef.sd(kk,:,:),a([3 4]))).*ysdmat;
    end
    tjef.sd=squeeze(tjef.sd);
 end

 sens.sa=sa;
 sens.totalMean=e0;
 sens.totalVar=vt;
 sens.smePm=smePm;
 sens.stePm=stePm;
 sens.mef=mef;
 sens.tmef=tmef;
 if ~isempty(varlist)
    sens.siePm=siePm;
    sens.jef=jef;
    sens.tjef=tjef;
 end
 if ~isempty(jelist), sens.sjePm=sjePm; end
end

function sa=componentSens(x,y,beta,lamUz,lamWs,xe,ngrid,varlist,jelist,rg,data)

% set-up
 diff=(rg(:,2)-rg(:,1))';
 [nmcmc,p]=size(beta); m=size(x,1);
 if isfield(data,'catInd')
   catInd=data.catInd;
 else
   catInd=zeros(1,p);
 end
 xdist=genDist(x,catInd);
 for ii=1:p
    xexdist(ii)=genDist2(xe(:,ii),x(:,ii),catInd(ii));
    xedist(ii)=genDist(xe(:,ii),catInd(ii));
 end
 if ~isempty(varlist)
    for ii=1:size(varlist,1)
       [xte1,xte2]=ndgrid(xe(:,varlist(ii,1)),xe(:,varlist(ii,2)));
       xte=[xte1(:) xte2(:)];
       xexdist(p+ii)=genDist2(xte,x(:,varlist(ii,:)),[catInd(varlist(ii,:))]);
       xedist(p+ii)=genDist(xte,[catInd(varlist(ii,:))]);
    end
 end
 P=zeros(nmcmc,m,m);
 Q=zeros(nmcmc,m,m);
 My=zeros(m,nmcmc);

 for ii=1:nmcmc
    betaei=beta(ii,:);
    lamUzi=lamUz(ii);
    lamWsi=lamWs(ii);

    % eta cov for the data & prediction locations
    S=gCovMat(xdist,betaei',lamUzi,lamWsi);

    P(ii,:,:)=inv(S);
    My(:,ii)=S\y;
    Q(ii,:,:)=squeeze(P(ii,:,:))-My(:,ii)*My(:,ii)';
 end

 % compute variances and functions
 e0 = zeros(nmcmc,1);
 e2 = zeros(nmcmc,1);
 vt=zeros(nmcmc,1);
 sme=zeros([nmcmc p]);
 ste=zeros([nmcmc p]);
 mef.m=zeros([nmcmc p ngrid]); mef.v=zeros([nmcmc p ngrid]);
 u1=zeros(m,1); u2=zeros(m,1);
 if ~isempty(varlist)
    sie=zeros([nmcmc size(varlist,1)]);
    jef.m=zeros([nmcmc size(varlist,1) ngrid ngrid]);
    jef.v=zeros([nmcmc size(varlist,1) ngrid ngrid]);
    u3=zeros(m,1);
 end
 if ~isempty(jelist), sje=zeros([nmcmc length(jelist)]); end
 for ii=1:nmcmc
    betaei=beta(ii,:);
    lamUzi=lamUz(ii);
    lamWsi=lamWs(ii);
    % initial calculations
    c1=calc1(betaei,diff); C2=calc2(x,xdist,m,rg,betaei,diff);
    for jj=1:m, c3(jj,:)=calc3(x(jj,:),rg,betaei,diff); end
    u2=prod(c3,2);
    e2(ii)=prod(c1)/lamUzi-trace(squeeze(Q(ii,:,:))*...
           varf(m,p,[],C2,u2))/lamUzi^2;
    e0(ii)=u2'*My(:,ii)/lamUzi;
    % total variance
    vt(ii)=1/lamUzi-trace(squeeze(Q(ii,:,:))*...
           varf(m,p,1:p,C2,[]))/lamUzi^2-e2(ii);
    % main/total effect indices; main effect functions
    for jj=1:p
       Js=[jj]; ll=setxor(1:p,Js);
       u1=prod(c3(:,ll),2); u4=prod(c1(ll));
       sme(ii,jj)=u4/lamUzi-trace(squeeze(Q(ii,:,:))*...
                  varf(m,p,Js,C2,u1))/lamUzi^2-e2(ii);
       sme(ii,jj)=sme(ii,jj)/vt(ii);
       ME=etae(Js,x,u1,u4,xexdist(jj),xedist(jj),betaei,lamUzi,...
               lamWsi,My(:,ii),squeeze(P(ii,:,:)));
       mef.m(ii,jj,:)=ME.m; mef.v(ii,jj,:)=ME.v;
       ll=[jj]; Js=setxor(1:p,ll); u2=prod(c3(:,ll),2);
       ste(ii,jj)=c1(ll)/lamUzi-trace(squeeze(Q(ii,:,:))*...
                  varf(m,p,Js,C2,u2))/lamUzi^2-e2(ii);
       ste(ii,jj)=1-ste(ii,jj)/vt(ii);
    end
    % two-factor interaction indices, joint effects
    if ~isempty(varlist)
       for jj=1:size(varlist,1)
          Js=varlist(jj,:); ll=setxor(1:p,Js);
          u3=prod(c3(:,ll),2); u5=prod(c1(ll));
          sie(ii,jj)=u5/lamUzi-trace(squeeze(Q(ii,:,:))*...
                     varf(m,p,Js,C2,u3))/lamUzi^2-e2(ii);
          sie(ii,jj)=sie(ii,jj)/vt(ii)-sme(ii,varlist(jj,1))-...
                     sme(ii,varlist(jj,2));
          JE=etae(Js,x,u3,u5,xexdist(p+jj),xedist(p+jj),betaei,lamUzi,...
                  lamWsi,My(:,ii),squeeze(P(ii,:,:)));
          jef.m(ii,jj,:,:)=reshape(JE.m,ngrid,ngrid);
          jef.v(ii,jj,:,:)=reshape(JE.v,ngrid,ngrid);
       end
    end
    % joint effect indices
    if ~isempty(jelist)
       for jj=1:length(jelist)
          Js=jelist{jj}; ll=setxor(1:p,Js);
          u6=prod(c3(:,ll),2); u7=prod(c1(ll));
          sje(ii,jj)=u7/lamUzi-trace(squeeze(Q(ii,:,:))*...
                     varf(m,p,Js,C2,u6))/lamUzi^2-e2(ii);
          sje(ii,jj)=sje(ii,jj)/vt(ii);
       end
    end
 end

 sa.e0=e0;
 sa.vt=vt;
 sa.sme=sme;
 sa.ste=ste;
 if ~isempty(varlist), sa.sie=sie; end
 sa.mef=mef;
 if ~isempty(varlist), sa.jef=jef; end
 if ~isempty(jelist), sa.sje=sje; end
end

function sc1=scalc1(beta,tg,Js,Is,theta,catInd)
 sc1=zeros(ntg,ntg,nmcmc);
 for ii=1:ntg
  xt1=theta(Is(:,ii),:); xt1(:,Js)=tg(ii);
  for jj=ii:ntg
   xt2=theta(Is(:,jj),:); xt2(:,Js)=tg(jj);
   xt12dist=genDist2(xt1,xt2,catInd);
   sc1(ii,jj,:)=mean(exp(-xt12dist.d*beta),1);
  end
 end
end

function sc2=scalc2(x,beta,tg,Js,Is,theta,catInd)
 sc2=zeros(ntg,m,nmcmc);
 for ii=1:ntg
  xt1=theta(Is(:,ii),:); xt1(:,Js)=tg(ii);
  for jj=1:m
   xt2=x(jj,:);
   xt12dist=genDist2(xt1,xt2,catInd);
   sc2(ii,jj,:)=mean(exp(-xt12dist.d*beta),1);
  end
 end
end

function c1=calc1(beta,diff)
 c1=zeros(1,length(beta));
 c1=(sqrt(pi./beta).*diff.*(2*normcdf(sqrt(2*beta).*diff)-1)-...
    (1./beta).*(1-sqrt(2*pi)*normpdf(sqrt(2*beta).*diff)))./...
    (diff.^2);
end

function C2=calc2(x,xdist,m,rg,beta,diff)
 kk=0; C2=zeros(m*(m+1)/2,length(beta));
 for ii=1:m-1
  for jj=ii+1:m
   kk=kk+1;
   mp=(x(ii,:)+x(jj,:))/2; di=find(xdist.indi==ii & xdist.indj==jj);
   C2(kk,:)=calc3(mp,rg,2.*beta,diff).*exp(-beta.*xdist.d(di,:)/2);
  end
 end
 for ii=1:m
  kk=kk+1;
  C2(kk,:)=calc3(x(ii,:),rg,2.*beta,diff);
 end
end

function c3=calc3(x,rg,beta,diff)
 c3=zeros(1,length(beta));
 c3=sqrt(pi./beta).*(normcdf(sqrt(2*beta).*(rg(:,2)'-x))-...
    normcdf(sqrt(2*beta).*(rg(:,1)'-x)))./diff;
end

function Vf=varf(m,p,Js,C2,ef)
 kk=0; ll=setxor(1:p,Js); Vf=zeros(m,m);
 for ii=1:m-1
  for jj=ii+1:m
   kk=kk+1; Vf(ii,jj)=1;
   if ~isempty(Js), Vf(ii,jj)=prod(C2(kk,Js)); end
   if ~isempty(ll), Vf(ii,jj)=Vf(ii,jj)*ef(ii)*ef(jj); end
  end
 end
 Vf=Vf+Vf';
 for ii=1:m
  kk=kk+1; Vf(ii,ii)=1;
  if ~isempty(Js), Vf(ii,ii)=prod(C2(kk,Js)); end
  if ~isempty(ll), Vf(ii,ii)=Vf(ii,ii)*(ef(ii)^2); end
 end
end

function ee = etae(Js,x,ef,vf,xexdist,xedist,beta,lamUz,lamWs,My,P)
 nxe=xedist.n;
 ee.m=zeros(nxe,1); ee.v=zeros(nxe,1);
 Ct=gCovMat(xexdist,beta(Js)',lamUz); 
 Ct=Ct.*repmat(ef',nxe,1);
 ee.m=Ct*My; 
 C=gCovMat(xedist,beta(Js)',lamUz,lamWs);
 ee.v=diag(C.*vf-Ct*P*Ct');
end
