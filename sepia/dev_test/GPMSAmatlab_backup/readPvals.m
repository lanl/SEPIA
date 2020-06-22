function pvals=readPvals(filename)

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

