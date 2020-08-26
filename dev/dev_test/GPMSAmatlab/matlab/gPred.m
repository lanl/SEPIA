%function pred=gPred(xpred,pvals,model,data,mode,theta)
%  Predict using a gpmsa constructed model. 
%  this is an interface to the new gPredict for backward compatibility

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pred=gPred(xpred,pvals,model,data,mode,theta)

  if strcmp(mode,'etamod');
    mode='wpred';
    theta=[];
  end

  if exist('theta','var'); 
    pred=gPredict(xpred,pvals,model,data,'mode',mode,'theta',theta,'returnMuSigma',1);
  else
    pred=gPredict(xpred,pvals,model,data,'mode',mode,'returnMuSigma',1);
  end

end