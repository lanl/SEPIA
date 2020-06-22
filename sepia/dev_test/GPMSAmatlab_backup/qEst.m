function th=qEst(pout,pvec,thProb, densFun, varargin) 
% function th=subRegionSamp(pout,pvec,thProb, densFun, varargin) 
  % collect response MLpost, into sets H, M, L, based on the
  % vl and vh estimates. Estimate response from M set, given integrated
  % density from L and H.
  %  Operations are generally defined on the native scale, requiring the
  %  simData.orig.xmin and simData.orig.xrange structures to be in place
  % Arguments:
  %   pvec - candidates for random draws of parameter sets
  %   thProb - threshold probability
  %   densFun - density function handle, to get probability of draw from X.
  %       The density function computes a density of each row vector in the
  %       matrix it is called with. It operates on the native scale space.
  % Optional arguments:
  %   poolSamp - the number of samples in the vl estimation pool; def. 1e6
  %   drawNewPoints - default 1; if false load previously saved vl est. pool
  %   savePool - default 0; if true, saved vl est. pool to 'estPoolDraws'
  %   numDraws - the number of threshold draws to perform; default 10
  %   rlzSamp - the sample size of realizations; default 500
  %   intMin, intMax - the boundaries of integration. Default is [0,1] in
  %                    each dimension. Scale is native space.
  %   GPest - default false; true indicates that the samples from the M set
  %          should be modeled by a GP and oversampled from the mean 
  %   GPestSamp - number of samples from M to generate per realization
  %          (default 1e5)
  %   doPlot, doPlot2 - default false, perform diagnostic plots (see code)
  %   saveFile - name of samples datafile, default 'estPoolDraws'
  
  numDraws=10;
  rlzSamp=500;
  poolSamp=1e6;
  numVars=pout.model.p;
  cubeMin=zeros(1,numVars);
  cubeMax=ones(1,numVars);
  drawNewPoints=1;
  savePool=0;
  GPest=0; GPestSamp=1e5;
  doPlot1=0; doPlot2=0;
  saveFile='estPoolDraws';
  k1=2;
  k2=3;
  parseAssignVarargs({'numDraws','poolSamp','rlzSamp','cubeMin','cubeMax', ...
                      'drawNewPoints','savePool','GPest','GPestSamp',...
                      'doPlot1','doPlot2','saveFile','k1','k2'});
                    
  % backwards compat for categorical capability
  if ~isfield(pout.data,'catInd'); pout.data.catInd=[]; end
  
    pvals=pout.pvals(pvec);
    
    % set up the sampling region [offset range]
      sampCube.range=cubeMax(:)'-cubeMin(:)'; 
      sampCube.offset=cubeMin(:)';
      
    % mapping from the sampling cube to the scaled x prediction area
    % d starts in [0,1], projects to sampCube in native scale, then back to
    % the prediction scaling.
    % native = d*scrange+scoffset
    % scaled = (native-pcoffset)/pcrange
    %        = (d*scrange+scoffset - pcoffset)/pcrange
    %        = d*scrange/pcrange + (scoffset-pcoffset)/pcrange
      predMap.range=sampCube.range ./ pout.simData.orig.xrange;
      predMap.offset=(sampCube.offset - pout.simData.orig.xmin) ...
                      ./ pout.simData.orig.xrange;
      
    % find the vl and vh estimates regions
    if drawNewPoints
        % get new points, and put them in the specified interval in scaled
        % space.
        d=rand(poolSamp,numVars) .* ...
            repmat(predMap.range,poolSamp,1) + ...
            repmat(predMap.offset,poolSamp,1);
        % prob function is in native scale
        p=densFun(d.*repmat(pout.simData.orig.xrange,poolSamp,1) + ...
                     repmat(pout.simData.orig.xmin,poolSamp,1) );
        % pred is, of course, in scaled space
        [r rs]=predictPointwise(pout,d,'mode','MAP');
        % Normalize probability - outmoded
        % p=p*prod(pout.simData.orig.xrange.*sampCube.range);
        [v  sri ]=calcRespThresh(p,r,thProb);
        [vl sril]=calcRespThresh(p,r-k1*rs,thProb);
        [vh srih]=calcRespThresh(p,r+k1*rs,thProb);
        fprintf('MAP thresh=%f\n',v);
        % break into 3 sets.
          LFlag= (r+k2*rs)<vl;
          UFlag= (r-k2*rs)>vh;
          Ld=d(LFlag,:);  Ud=d(UFlag,:);  Md=d(~(LFlag|UFlag),:);
          Lr=r(LFlag);    Ur=r(UFlag);    Mr=r(~(LFlag|UFlag));
          Lp=p(LFlag);    Up=p(UFlag);    Mp=p(~(LFlag|UFlag));
          Lprob=sum(Lp)/sum(p);
          Uprob=sum(Up)/sum(p);
        if savePool
          fprintf('Saving new ML mean draws\n');
          save(saveFile,'d','p','r','rs','v','sri','vl','sril','vh','srih',...
                        'thProb','poolSamp','LFlag','UFlag','Ld','Ud','Md',...
                        'Lr','Ur','Mr','Lp','Up','Mp','Lprob','Uprob') ;
        end
    else
        fprintf('Loading saved ML mean draws\n');
        load(saveFile);
        fprintf('Loaded %d draws and threshold %f calculated sets \n',poolSamp,thProb);        
    end

    fprintf('p(r<vl=%6.3f): %d elements with %f prob\n',vl,sum(LFlag),Lprob);
    fprintf('p(r>vh=%6.3f): %d elements with %f prob\n',vh,sum(UFlag),Uprob);
    
    if doPlot1
        clf;
        plot(cumsum(p(sri))/sum(p),r(sri), 'b'); hold on;
        plot(cumsum(p(sril))/sum(p),r(sril)-k1*rs(sril), 'r');
        plot(cumsum(p(srih))/sum(p),r(srih)+k1*rs(srih), 'r');
        a=axis;
        plot (thProb* [1 1], a ([3 4]),'k:') ;
        plot([0 1],vl*[1 1 ],'k:') ;
        plot([0 1],vh*[1 1 ],'k:');
        plot([0 1], v*[1 1 ],'k:') ;
    end
    if doPlot2
      clf; 
      plotxy(Ld(ilinspace(1,end,10000),:),'.'); 
      hold on; 
      plotxy(Md(ilinspace(1,end,1000),:),'g.'); 
      plotxy(Ud(ilinspace(1,end,2000),:),'r.')
    end
              
    % Purge M set of the draws that are irrelevant due to no prob. mass
     % we could mess up in the case of uniform distributions, throwing away
     % everything by mistake, so make sure we have a lot of values before
     % starting this
     if length(unique(Mp))/length(Mp) > 0.1
      Ms=sort(Mp,1,'descend');
      pThresh = Ms(find(cumsum(Ms)/sum(Ms) > 0.9999,1));
      Md=Md(Mp>pThresh,:);
      Mr=Mr(Mp>pThresh);
      Mp=Mp(Mp>pThresh);
     end

    % sample threshold with draws from the M-relevant set
    %counter('stime',l,numDraws,10,0);
    if size(Md,1)<rlzSamp; 
      error('\n not enough samples for realization sample size \n');
    end
    if numDraws; th(numDraws).th=[]; end
    for ii=1:numDraws
        %counter(ii);
        tic;
        samp=gSample(size(Md,1),rlzSamp);
        pred=gPredict(Md(samp, :),pvals(ceil(rand*end)), ...
            pout.model,pout.data,'addResidVar',1,'returnMuSigma',1); 
        if GPest
          % set up a new model that includes the new predictions in M
          %   (which was a realization in M)
          fprintf('Drawing from GP mean ...')
          pouts=pout;
          pouts.model.m = pouts.model.m+rlzSamp;
          pouts.model.w =[pouts.model.w;pred.Myhat'];
          pouts.data.w=pouts.model.w;
          pouts.data.zt=[pouts.data.zt; Md(samp,:)];
          pouts.model.ztDist=genDist(pouts.data.zt,pouts.data.catInd);
          
          % predict a whole lot of points, and keep whatever fraction is in
          % M -- should it loop until a minimum number are found? 
            Mresp=[]; Mprob=[]; Mdes=[];
            while length(Mresp)<GPestSamp
              GPd=rand(GPestSamp,numVars) .* ...
                   repmat(predMap.range,GPestSamp,1) + ...
                   repmat(predMap.offset,GPestSamp,1);
              [GPr GPrs]=predictPointwise(pouts,GPd,'toNative',0);
              GPLFlag= (GPr+3*GPrs)<vl;
              GPUFlag= (GPr-3*GPrs)>vh;
              Mresp=[Mresp; GPr(~(GPLFlag|GPUFlag))];
              GPMd=GPd(~(GPLFlag|GPUFlag),:);
              GPMdNum=size(GPMd,1);
              Mprob=[Mprob; 
                densFun(GPMd.*repmat(pout.simData.orig.xrange,GPMdNum,1) + ...
                             repmat(pout.simData.orig.xmin,GPMdNum,1) ) ];
              Mdes=[Mdes;GPMd];
            end
            probs=[Lprob; (1-Lprob-Uprob)* Mprob/sum(Mprob); Uprob];
            yhat=Mresp*pout.simData.orig.ysd + pout.simData.orig.ymean;
            hats=[-Inf; yhat; Inf];
        else
          ypr=squeeze(pred.w);
          yhat=ypr*pout.simData.orig.ysd + pout.simData.orig.ymean;
          probs=[Lprob; (1-Lprob-Uprob)* Mp(samp)/sum(Mp(samp)); Uprob];
          hats=[-Inf; yhat; Inf];
        end
        th(ii).th=calcRespThresh(probs,hats,thProb);
        th(ii).probs=probs;
        th(ii).yhat=hats;
        if GPest
          th(ii).Mdes=Mdes;
        end
        fprintf('Draw %2d threshold %f; took %4.1fs\n',ii,th(ii).th,toc);
    end
    %counter ( 'end' )
end

function [th sri]=calcRespThresh(p,r,thProb)
  % do the magic
  [sr sri]=sort(r);
  syp=p(sri);
  sypc=cumsum(syp)/sum(syp);
  pthi=find( (sypc-thProb)>0 , 1);
  % response cutoff between sypc(pthi-1) and sypc(pthi)
  % linear interp
  th=interp1(sypc([pthi-1 pthi]),sr([pthi-1 pthi]),thProb,'linear');
end

function [r rs]=predictPointwise(pout,xp,varargin)

  mode='MAP';
  pvec=1:length(pout.pvals);
  toNative=1;
  verbose=0;
  parseAssignVarargs({'mode','pvec','toNative','verbose'});

  switch(mode)
    case 'MAP'
      % get the most likely model
      pvals=pout.pvals;
      lp=[pvals.logPost]; [lpm lpi]=max(lp);
      pvalM=pvals(lpi);
    case 'Mean'
      % get the mean model
      vars={'betaU','lamUz','lamWs','lamWOs'};
      for ii=1:length(vars)
        pvalM.(vars{ii})=mean([pout.pvals(pvec).(vars{ii})],2);
      end
  end


  % get the response from that model
  chunk=50;
  numSamp=size(xp,1);
  if (numSamp/chunk)~=round(numSamp/chunk)
      error('samples not a multiple of chunk (=%d)\n',chunk);
  end
  xp(xp<0)=0; xp(xp>1)=1;
  r=zeros(numSamp,1);
  rs=r;
  SigDataInv=computeSigDataInv(pout,pvalM);
  if verbose; fprintf('predictModML: predicting\n'); end
  if verbose; counter('stime',1,numSamp,10,6); end
  for ii=1:numSamp/chunk;
      if verbose; counter(ii*chunk); end
      x=xp((ii-1)*chunk +1:ii*chunk, :);
      pred=gPredLocal(x,pout,pvalM,SigDataInv);
      yhs=squeeze(pred.Myhat);
      shs=sqrt(diag(pred.Syhat));
      if toNative 
        r((ii-1)*chunk +1:ii*chunk)=...
          yhs*pout.simData.orig.ysd+pout.simData.orig.ymean;
        rs((ii-1)*chunk +1:ii*chunk)=shs*pout.simData.orig.ysd;
      else
        r((ii-1)*chunk +1:ii*chunk)=yhs;
        rs((ii-1)*chunk +1:ii*chunk)=shs;
      end
  end
  if verbose; counter ( 'end' ) ; end
end

function SigDataInv=computeSigDataInv(pout,pvals)
    model=pout.model;
    m=model.m;p=model.p;q=model.q;pu=model.pu;
    betaU=reshape(pvals.betaU,p+q,pu);
    lamUz=pvals.lamUz;
    lamWs=pvals.lamWs; lamWOs=pvals.lamWOs;
    diags1=diagInds(m*pu);
    SigData=zeros(m*pu);
    for jj=1:pu
        bStart=(jj-1)*m+1; bEnd=bStart+m-1;
        SigData(bStart:bEnd,bStart:bEnd)=...
            gCovMat(model.ztDist,betaU(:,jj),lamUz(jj));
    end
    SigData(diags1)=SigData(diags1)+ ...
        kron(1./(model.LamSim*lamWOs)',ones(1,m)) + ...
        kron(1./(lamWs)',ones(1,m)) ;
    SigDataInv=inv(SigData);
end

function pred=gPredLocal(xpred,pout,pvals,SigDataInv)
    data=pout.data; model=pout.model;
    m=model.m;p=model.p;q=model.q;pu=model.pu;
    npred=size(xpred,1);

    diags2=diagInds(npred*pu);
    betaU=reshape(pvals.betaU,p+q,pu);
    lamUz=pvals.lamUz;
    lamWs=pvals.lamWs; lamWOs=pvals.lamWOs;
    xpredDist=genDist (xpred) ;
    zxpredDist=genDist2(data.zt,xpred);
    SigPred=zeros(npred*pu);
    for jj=1:pu
        bStart=(jj-1)*npred+1; bEnd=bStart+npred-1;
        SigPred(bStart:bEnd,bStart:bEnd)= ...
            gCovMat(xpredDist,betaU(:,jj),lamUz(jj));
    end
    SigPred(diags2)=SigPred(diags2)+ ...
        kron(1./(model.LamSim*lamWOs)',ones(1,npred)) + ... % resid var
        kron(1./(lamWs)',ones(1,npred)) ;
    SigCross=zeros(m*pu,npred*pu);
    for jj=1:pu
        bStartI=(jj-1)*m+1;      bEndI=bStartI+m-1;
        bStartJ=(jj-1) *npred+1; bEndJ=bStartJ+npred-1;
        SigCross(bStartI:bEndI,bStartJ:bEndJ)=...
            gCovMat(zxpredDist,betaU(:,jj),lamUz(jj));
    end
    % Get the stats for the prediction stuff.
    W=(SigCross')*SigDataInv;
    pred.Myhat=W*(data.w(:));
    pred.Syhat=SigPred-W*SigCross;
end



