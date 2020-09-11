function out=nedderimp(t1,params);
% gives inner radius as a function of time for a cylinder implosion
% input t1: spits out inner radius times at each of the times in vector t1.
% the 6-vector params = [R1 lam s rho mrat u0]
%
% function that computes the time vs. y (inner radius) trace of the implosion
% according to Neddermeyer's formula given in his '43 paper.
% The basic form of the equation is of the form:
%
% y' = [2/(R1^2*f(y)^2)*{v0^2/2 - s/(2*rho)*g(y)}]^{-.5}
% 
% where
% f(y) = (y^2/(1-lam^2))*ln((y^2+1-lam^2)/y^2)   and
% g(y) = 1/(1-lam^2)*[y^2+1-lam^2]*ln(y^2+1-lam^2)-lam^2*ln(lam^2)
%
% we're simulating experiment 10 in Neddermeyer's '43 paper.
% 
% lam = 2/3   ratio of outer cylinder radius to the inner radius
% s = 3*10^5*6.84*10^4  yield stress in cgs units (300,000 lb/in^2)
% rho = 7.5 g/cm^3  -- specific density (relative to water)
% R1 = 1.5in  -- initial outer radius of steel cylinder being imploded
% R2 = 1.0in  -- initial inner radius of steel cylinder being imploded
% v0 = .3*10^5  cm/s -- initial velicity imparted on outer radius of
%                       cylinder from the HE
% mrat = mass ratio of HE to cylinder = rho_HE*h_HE/(rho_cyl*h_cyl) = .32
%        in expt 10
% u0 = energy per gram of exploded gas from the HE
% critical velocity given by sqrt(g(0)*s/rho) = 5.816*10^4 cm/s
% final scaled inner radius given by solving
%  v0^2/2 - s/(2*rho)*g(y)=0  => resluting y = .505
% y = r2/R1; x = r1/R1; x^2 - y^2 = lam^2 (conservation of mass)
%
% R1 = initial outer radius;  r1 = outer radius as a function of time...
% R2 = initial inner radius;  r2 = inner radius as function of time...
% lam = R2/R1 = 1.0in/1.5in = 2/3.

R1 = params(1); lam = params(2); R2 = R1*lam; 
s = params(3); rho = params(4); mrat = params(5); u0 = params(6);
v0 = mrat*sqrt(2*u0/(1+mrat));
% the 6-vector params = [R1 lam s rho mrat u0]
nt = length(t1);
dt = 1e-8;
ninc = 8000;
yout = zeros([ninc 1]);
yout(1) = lam;
tout = (0:dt:dt*(ninc-1))';
for i=2:ninc
    yout(i) = yout(i-1) + (v(yout(i-1),params)*dt);
end
ineg = (yout < 0);
yout(ineg) = 0;

out = interp1(tout,yout,t1,'linear');


function fout = f(y,params) % computes Neddermeyer's f(y)
R1 = params(1); lam = params(2); R2 = R1*lam; 
s = params(3); rho = params(4); mrat = params(5); u0 = params(6);
v0 = mrat*sqrt(2*u0/(1+mrat));
fout = y.^2/(1-lam^2).*log((y.^2 + 1 - lam^2)./y.^2);

function gout = g(y,params) % computes Neddermeyer's g(y)
R1 = params(1); lam = params(2); R2 = R1*lam; 
s = params(3); rho = params(4); mrat = params(5); u0 = params(6);
v0 = mrat*sqrt(2*u0/(1+mrat));
gout = 1/(1-lam^2).*(y.^2.*log(y.^2)-(y.^2+1-lam^2).*log(y.^2+1-lam^2)-lam^2*log(lam^2));

function vout = v(y,params) % computes the velocity as a function of y
R1 = params(1); lam = params(2); R2 = R1*lam; 
s = params(3); rho = params(4); mrat = params(5); u0 = params(6);
v0 = mrat*sqrt(2*u0/(1+mrat));
x1 = v0^2/2 - s.*g(y,params)/(2*rho);
ineg = x1 < 0;
x1(ineg) = 0;
vout = -sqrt(2./(R1^2*f(y,params)).*x1);
