function writeModel(params,filename)
%function writeModel(params,filename)
% This writes out the data for a GPM/SA model, to be read by the C version
% of the likelihood and sampling code.

n=params.model.n;   % number of observations
m=params.model.m;   % number of simulations
p=params.model.p;   % number of observation independent variables
q=params.model.q;   % number of additional simulator independent variables
pv=params.model.pv; % size of the discrepancy (delta) basis dependent var.
pu=params.model.pu; % size of the response (eta) basis transformed dep. var.

fp=fopen(filename,'w');
fprintf(fp,'%d %d %d %d %d %d\n',n,m,p,q,pv,pu);

% write the observation independent variables
fprintf(fp,'%f ',params.data.x);
fprintf(fp,'\n');

% write the simulator independent variables
fprintf(fp,'%f ',params.data.zt);
fprintf(fp,'\n');

% write the observation discrepancy and eta response
fprintf(fp,'%f ',params.model.vu);
fprintf(fp,'\n');

% write the simulator eta response
fprintf(fp,'%f ',params.model.w);
fprintf(fp,'\n');

% initial values for lamOs model parameter(s)
fprintf(fp,'%f ',params.model.lamOs);
fprintf(fp,'\n');

% initial values for lamWOs model parameter(s)
fprintf(fp,'%f ',params.model.lamWOs);
fprintf(fp,'\n');

% initial values for theta parameters(s)
fprintf(fp,'%f ',params.model.theta);
fprintf(fp,'\n');

% initial values for betaV model parameter(s)
fprintf(fp,'%f ',params.model.betaV);
fprintf(fp,'\n');

% initial values for lamVz model parameter(s)
fprintf(fp,'%f ',params.model.lamVz);
fprintf(fp,'\n');

% initial values for betaU model parameter(s)
fprintf(fp,'%f ',params.model.betaU);
fprintf(fp,'\n');

% initial values for lamUz model parameter(s)
fprintf(fp,'%f ',params.model.lamUz);
fprintf(fp,'\n');

% initial values for lamWs model parameter(s)
fprintf(fp,'%f ',params.model.lamWs);
fprintf(fp,'\n');

% initial values for LamSim model parameter
fprintf(fp,'%f ',params.model.LamSim);
fprintf(fp,'\n');

% initial values for SigObs model parameter
fprintf(fp,'%f ',params.model.SigObs);
fprintf(fp,'\n');


% write prior and MCMC parameters
if n>0 
  varsA={'theta' 'rhoV'  'rhoU'  'lamVz' 'lamUz' 'lamWs' 'lamWOs' 'lamOs'};
  varsB={'theta' 'betaV' 'betaU' 'lamVz' 'lamUz' 'lamWs' 'lamWOs' 'lamOs'};
else
  varsA={'rhoU'  'lamUz'  'lamWs'  'lamWOs'};
  varsB={'betaU' 'lamUz'  'lamWs'  'lamWOs'};
end
for ii=1:length(varsA)
  fprintf(fp,'%f ',params.mcmc.([varsA{ii} 'width'])(1));  % mcmc step
  fprintf(fp,'%f ',params.priors.(varsB{ii}).bLower);      % lower bound
  fprintf(fp,'%f ',params.priors.(varsB{ii}).bUpper);      % upper bound
  fprintf(fp,'%f ',params.priors.(varsA{ii}).params(1,:)); % prior params
  fprintf(fp,'\n');
end


fclose(fp);
  
  

