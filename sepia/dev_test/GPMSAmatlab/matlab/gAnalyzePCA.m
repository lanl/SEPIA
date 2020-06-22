%function a=gAnalyzePCA(y,y1)
% Some analysis of principle components of supplied datasets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
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
  
