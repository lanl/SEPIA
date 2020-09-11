function dataStruct=neddeg();

load neddTemp

% now the data come in 3 expts with ydat1 ydat2 ydat3 holding the
% data and timedat1 timedat2 timedat3 holding the times at which
% the r's were measured, and phidat holds the angles for each
% time slice of each dataset

%
%
% Data is now complete, build up the structures for GPM modeling.
%
%

% fill up the simulator object - simData - with stuff.
simData.orig.y = ye;      % matrix of neta x m simulation output; each column is a different sim run.
simData.x = Xlhs;  % m x p vector of input settings (x,theta)
simData.orig.ymean = ysimmean; % neta-vector that's the mean of the m sim outputs.
simData.orig.ysd = ysimsd;  % sd of sims
simData.yStd=ysimStd;

simData.Ksim = Ksimpy;
simData.orig.time = time; % the values of the time for the sim output in the image matrix format
simData.orig.phi = phi; % the values of the angle for the sim output in the image matrix format
simData.orig.timemat = timearr;
simData.orig.phimat = phiarr;

% fill up the obsData() list - each component corresponds to data from a
% particular experiment. n = # of experiments (here n=3).  Everything  must
% be input here.  radius and angle are used later to interpolate the eof.
n = 3; % # of experiments
obsData(1).orig.y = ydat1(:);   % observed radii at various time slices
obsData(2).orig.y = ydat2(:);
obsData(3).orig.y = ydat3(:);

% observed x,theta conditions (note theta values are irrelevant here - they're just place holders)
for k=1:n 
    obsData(k).x = [xdat(k)]; 
end    

obsData(1).orig.time = origtime1;
obsData(2).orig.time = origtime2;
obsData(3).orig.time = origtime3;
obsData(1).orig.phi =  origphi1;  % phi angle values for the measurements.
obsData(2).orig.phi = origphi2;
obsData(3).orig.phi = origphi3;

% compute simulator mean values simdat.ymean interpolated to the data values...
obsData(1).orig.ymean = origymean1;
obsData(2).orig.ymean = origymean2;
obsData(3).orig.ymean = origymean3;

% now compute the centered, scaled observed arrival times yStd
obsData(1).yStd = origyStd1;
obsData(2).yStd = origyStd2;
obsData(3).yStd = origyStd3;

% now compute the interpolated eof's for each dataset - held in the 
obsData(1).Kobs = Kobs1;
obsData(2).Kobs = Kobs2;
obsData(3).Kobs = Kobs3;

% compute the basis functions for the discrepancy function.  Each data set
obsData(1).Dobs = Dobs1;
obsData(2).Dobs = Dobs2;
obsData(3).Dobs = Dobs3;
for k=1:n
    obsData(k).orig.Dsim = Dsimpy;
end

for k=1:n
    obsData(k).orig.knotlocstime = knotlocstime;
    obsData(k).orig.knotlocsphi = knotlocsphi;
end

dataStruct.obsData=obsData;
dataStruct.simData=simData;