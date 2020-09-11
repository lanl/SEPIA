function dout=dist2pi(x1,x2);% computes the distance assuming periodicity: 2pi=0
% x1 and x2 are vectors with common length and values% between 0 and 2pid = abs(x1-x2);iwrap = d > pi;d(iwrap) = 2*pi - d(iwrap);dout = d;
	