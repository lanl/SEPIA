function out=dnorm(x,mu,scale);
%normal density in 1-d. 
%It is scaled so that the 1-d integral is 1
%mu and scale are scalars, x is an array...

out=zeros(size(x));
u=abs(x-mu)./scale;
out = 1.0/sqrt(2*pi)/scale * exp(-.5 * u.^2);
