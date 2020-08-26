function pvals=readPvals(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp=fopen(filename,'r');

n=fscanf(fp,'%d',1);
m=fscanf(fp,'%d',1);
p=fscanf(fp,'%d',1);
q=fscanf(fp,'%d',1);
pv=fscanf(fp,'%d',1);
pu=fscanf(fp,'%d',1);

if n==0; etaMod=1; else etaMod=0; end

pvi=1;
while ~feof(fp)
  if ~etaMod
    pvals(pvi).theta=fscanf(fp,'%f',q);
    pvals(pvi).betaV=fscanf(fp,'%f',1*p);
    pvals(pvi).lamVz=fscanf(fp,'%f',1);
    pvals(pvi).lamOs=fscanf(fp,'%f',1);
  end
  pvals(pvi).betaU=fscanf(fp,'%f',pu*(p+q));
  pvals(pvi).lamUz=fscanf(fp,'%f',pu);
  pvals(pvi).lamWs=fscanf(fp,'%f',pu);
  pvals(pvi).lamWOs=fscanf(fp,'%f',1);
  pvals(pvi).logLik=fscanf(fp,'%f',1);
  pvals(pvi).logPrior=fscanf(fp,'%f',1);
  pvals(pvi).logPost=pvals(pvi).logLik+pvals(pvi).logPrior;
  pvi=pvi+1;
end
pvals(end)=[];

fp=fclose(fp);

