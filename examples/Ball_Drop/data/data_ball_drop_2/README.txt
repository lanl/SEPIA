The data files hold simulations and data for various ball drop examples

Field data for the 4 different balls; 3 observations each, and
6 observations for the bowling ball
% R_ball rho_air rho_ball height time sd(time); 1st 3 are basketball
% drops, 2nd 3 are baseball drops, 3rd 6 are bowling ball drops, the 4th 3
% are softball drops.
% See towereg.m (for example to see how data are read in)
fieldDat15x6gparam.txt

I have made fieldDat12x6 that has the info for the 3 bowling ball drops at 20,40,60

	
% design over R_ball rho_air rho_ball C_D and g used for example where
% g and C_D are to be estimated with all balls (except softball)
% a common 20-run design varying g and C_D is carried out for each
% of the 4 balls, giving the first 80 runs in the design set.  The
% values of R_ball rho_air rho_ball are set to each experiment. The
% final 60 runs are a space-filling LHS over the 5-d input space.
% For the runs, we’ll only use the first 80 runs from this design.
% 
desNative140x5gCparam.txt


% sims from 140 run design varying g and C_D; here the standard model
% for drag is used from drop3.m
sims101x140gCparam.out
% sims from 140 run design varying g and C_D; here the non-standard model

% heights for which the drop times are computed.
simHeights101x1
