%function a=gAnalyzePCA(y,y1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
%
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
%
% Copyright 2008.  Los Alamos National Security, LLC. This material 
% was produced under U.S. Government contract DE-AC52-06NA25396 for 
% Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
% National Security, LLC for the U.S. Department of Energy. The U.S. 
% Government has rights to use, reproduce, and distribute this software.  
% NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY 
% WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF 
% THIS SOFTWARE.  If software is modified to produce derivative works, 
% such modified software should be clearly marked, so as not to confuse 
% it with the version available from LANL.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a K]=gAnalyzePCA(y,y1)

[U S V]=svd(y,0);

K = (U*S)/sqrt(size(y,2));
a=cumsum(diag(S).^2); a=a/a(end);

figure(10); clf;
  subplot(1,2,1); hold on; plot(a); axis([1 length(a) 0 1])
  subplot(1,2,2); hold on; plot(a(1:min(end,10))); axis([1 10 0 1])  

% add the mean absolute deviation of the simulators
if exist('y1')
  for ii=1:size(K,1); 
    y1pcv(ii)=sum(abs(y1-K(:,1:ii)*(K(:,1:ii)\y1))); 
    ypcv(ii)=sum(sum(abs(y-K(:,1:ii)*(K(:,1:ii)\y))));
  end
  y1pcv=1-y1pcv/sum(abs(y1));
  ypcv=1-ypcv/sum(abs(y(:)));
  subplot(1,2,1); plot(ypcv,'g'); plot(y1pcv,'r');
  legend({'variance explained','sim abs resid explained','obs abs resid explained'},'location','Best');
  subplot(1,2,2); plot(ypcv(1:min(end,10)),'g'); plot(y1pcv(1:min(end,10)),'r'); 
  legend({'variance explained','sim abs resid explained','obs abs resid explained'},'location','Best');
  title('zoom on first 10 PCs')
end
  
figure(11); clf
  PC=U*S;
  for ii=1:5; 
    h(ii)=subplot(5,1,ii); 
      plot(PC(:,ii)); 
      title(['PC ' num2str(ii)]);
  end
  axisNorm(h,'xymax');
  
  figure(12); clf;
   for ii=1:10; 
     h(ii)=subplot(10,1,ii); 
     K=U(:,1:ii)*S(1:ii,1:ii);
     pc=K\y;
     yhat=K*pc;
     plot(yhat-y); 
     title(['reconstruction error with ' num2str(ii) ' PC']);
   end;
   axisNorm(h,'ymax')
  
