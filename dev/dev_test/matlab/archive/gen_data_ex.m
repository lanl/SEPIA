% Author: James R. Gattiker, Los Alamos National Laboratory
%
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079
%
% � Copyright Los Alamos National Security, LLC.
%
% This Software was produced under a U.S. Government contract (DE-AC52-06NA25396) 
% by Los Alamos National Laboratory, which is operated by the Los Alamos National 
% Security, LLC (LANS) for the U.S. Department of Energy, National Nuclear Security 
% Administration. The U.S. Government is licensed to use, reproduce, and distribute 
% this Software. Permission is granted to the public to copy and use this Software 
% without charge, provided that this Notice and any statement of authorship are 
% reproduced on all copies. Neither the Government nor the LANS makes any warranty, 
% express or implied, or assumes any liability or responsibility for the user of
% this Software. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data=gen_data_ex(n_pc)

%reset the rand function for reproducible results
  rand('state',0); 
  
%%%% Create Data -  this is completely application specific. 
  function y=simFunc(x1,x2,time) % simulation response function def.
    y=(kron(x1,time.^2) + kron(x2,(time-0.5).^2))';
  end
  function y=disFunc(x1,x2,time); % obs to sim discrepancy function def.
    y=(kron((x1-0.5),sin(10*time)) .* kron(x2,exp(-3*time)) )';
  end
  % build a random (z,theta) design matrix with m examples
    % a little complicated, create a z grid, then get samples on each
      mz=20; mt=5;
      m=mz*mt; z=repmat(linspace(0,1,mz)',mt,1); t=sample(z,m);
  % build up y signals on a time axis
    ysimtime=0:0.05:1;
    ysim= simFunc(z,t,ysimtime); 
  % set up some observed data point locations, with a single theta
    n=3;
    x=[1:n]'/(n+1);  % brings the ends in a tick
    theta0=0.3; % true theta
  % build obs using a unique measurement grid for each
    for ii=1:n
      ttime=cumsum(rand(100,1)*0.1);
      yobs(ii).time=ttime(find(ttime<1)); % a random-ish time grid
      yobs(ii).y= (simFunc(x(ii),theta0,yobs(ii).time) + ...
                   disFunc(x(ii),theta0,yobs(ii).time) )';
      yobs(ii).Lamy=eye(length(yobs(ii).time)); % null corr in this case
    end

    %keyboard
    
%%%% Process the "raw" data into GPM/SA format

% 'x's and 'theta's are already standardized to [0,1] in this case; 
% in general, user must standardize x and theta to this range

% Standardize y to mean 0 var 1
  % simulations first
    ysimmean=mean(ysim,2);
    ysimStd=(ysim-repmat(ysimmean,1,m));
    ysimsd=std(ysimStd(:));
    ysimStd = ysimStd/ysimsd;
  % interpolate to data grid and standardize experimental data
    for ii=1:n
      yobs(ii).ymean=interp1(ysimtime,ysimmean,yobs(ii).time);
      yobs(ii).yStd=(yobs(ii).y-yobs(ii).ymean)/ysimsd;
    end

% K basis
  % compute on simulations
    pu=n_pc;
    [U,S,V]=svd(ysimStd,0);
    Ksim=U(:,1:pu)*S(1:pu,1:pu)./sqrt(m);
  % interpolate K onto data grids
    for ii=1:n
      yobs(ii).Kobs=zeros(length(yobs(ii).yStd),pu);
      for jj=1:pu
        yobs(ii).Kobs(:,jj)=interp1(ysimtime,Ksim(:,jj),yobs(ii).time);
      end
    end

% D basis
  % lay it out, and record decomposition on sim and data grids
    % Kernel centers and widths
      Dgrid=0.1:0.2:0.9; Dwidth=0.1; 
      pv=length(Dgrid);
    % Compute the kernel function map, for each kernel
      Dsim=zeros(size(ysimStd,1),pv);
      for ii=1:n; yobs(ii).Dobs=zeros(length(yobs(ii).yStd),pv); end
      for jj=1:pv
        % first the obs
          for ii=1:n
            yobs(ii).Dobs(:,jj)=normpdf(yobs(ii).time,Dgrid(jj),Dwidth);
          end
        % now the sim
          Dsim(:,jj)=normpdf(ysimtime,Dgrid(jj),Dwidth);
      end
    % normalize the D maps
      Dmax=max(max(Dsim*Dsim'));
      Dsim=Dsim/sqrt(Dmax);
      for ii=1:n; yobs(ii).Dobs=yobs(ii).Dobs/sqrt(Dmax); end

% record the data into the obsData and simData structures.
  % First simData
    % required fields
      simData.x   =[z t];
      simData.yStd=ysimStd;
      simData.Ksim=Ksim;
    % extra fields: original data and transform stuff
      simData.orig.y=ysim;
      simData.orig.ymean=ysimmean;
      simData.orig.ysd=ysimsd;
      simData.orig.Dsim=Dsim;
      simData.orig.time=ysimtime;
  % obsData
    for ii=1:n
      % required fields
        obsData(ii).x   =x(ii);
        obsData(ii).yStd=yobs(ii).yStd;
        obsData(ii).Kobs=yobs(ii).Kobs;
        obsData(ii).Dobs=yobs(ii).Dobs;
        obsData(ii).Lamy=yobs(ii).Lamy; % null Lamy in this case
      % extra fields
        obsData(ii).orig.y=yobs(ii).y;
        obsData(ii).orig.ymean=yobs(ii).ymean;
        obsData(ii).orig.time =yobs(ii).time;
    end

% pack up and leave
data.simData=simData;
data.obsData=obsData;

%dataPlots(simData,obsData);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dataPlots(simData,obsData)

 
 % Construct the v and the u
  for ii=1:length(obsData)
   pv=size(obsData(ii).Dobs,2); pu=size(obsData(ii).Kobs,2);
   DK=[obsData(ii).Dobs obsData(ii).Kobs];
   Sigy=inv(obsData(ii).Lamy);
   vu=inv(DK'*Sigy*DK+eye(pu+pv)*1e-4)*DK'*Sigy*obsData(ii).yStd;    
   v(:,ii)=vu(1:pv);
   u(:,ii)=vu(pv+1:end);
  end
 % construct the w
  w=simData.Ksim\simData.yStd;

 figure(1); clf % the observed data, decomposition, and reconstruction
  for ii=1:length(obsData)
    subplot(length(obsData),1,ii)
     % the original data (standardized)
     plot(obsData(ii).orig.time,obsData(ii).yStd,'k:') 
     hold on; 
     % the simulation component
     plot(obsData(ii).orig.time, ...
          obsData(ii).Kobs*u(:,ii),'r--')
     % the discrepancy component
     plot(obsData(ii).orig.time, ...
          obsData(ii).Dobs*v(:,ii),'g')
     % the reconstruction
     plot(obsData(ii).orig.time, ...
          obsData(ii).Kobs*u(:,ii) + ...
          obsData(ii).Dobs*v(:,ii),'b.')
     title(['Obs ' num2str(ii) ' x=' num2str(obsData(ii).x) ...
            ', orig, recon, Ku, and Dv']);
  end

 figure(2); clf % simulation data and residual error from reconstruction
  subplot(2,1,1) % first, the standardized data
    plot(simData.orig.time,simData.yStd);
    title('Original simulation data (scaled)');
  subplot(2,1,2) % then the residual
    plot(simData.orig.time,simData.yStd-simData.Ksim*w)
    title('Residual from reconstruction (scaled)');

figure(3); clf % the transform functions themselves
  subplot(2,1,1) % plot the principal components used
    plot(simData.orig.time,simData.Ksim)
    title('Principal Components from K ');
  subplot(2,1,2) % plot the kernel space
    plot(simData.orig.time,simData.orig.Dsim)
    title('Kernels from D');

drawnow;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function samp=sample(set,num)
  %same function as trandsample in the stats toolbox. returns num samples
  % set, without replacement.
  [y i]=sort(rand(length(set),1));
  samp=set(i(1:num));
end

