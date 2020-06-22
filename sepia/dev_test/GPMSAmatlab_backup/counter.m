% function counter('start',first_value,last_value,skip_counts,feed)
% function counter('stime',first_value,last_value,skip_seconds,feed)
%   Setup mode
%       first_value=first value in counter
%       last_value=last value in counter (for time computation)
%       feed = count of display events for computing time remaining (and linefeed)
%       skip_counts = count to skip any printout
%       skip_seconds = time delay for progress display
% function counter(index)
%   Run Mode
%       index = the current loop index
% function counter('end')
%       print the final time

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

function out=counter(arg1,arg2,arg3,arg4,arg5)

persistent first last feed lval skip mode dcount;

out=0;

if strcmp(arg1,'start')
    tic;
    mode=0;
    first=arg2; last=arg3; skip=arg4; feed=arg5;
    lval=first-skip;
    dcount=0;
    fprintf('Started value counter, vals %d -> %d\n',first,last);
    fprintf('        ');
    return;
elseif strcmp(arg1,'stime')
    tic;
    mode=1;
    first=arg2; last=arg3;
    skip=arg4;  feed=arg5;
    lval=0;
    dcount=0;
    fprintf('Started timed counter, vals %d -> %d\n',first,last);
    fprintf('        ');
    return;
elseif strcmp(arg1,'end')
    etime=toc;
    if etime>60; fprintf('%dmin:',floor(toc/60)); end;
    fprintf('%5.2fsec \n',mod(toc,60));
    return;
end    

val=arg1;

switch(mode)
case 0
    i=val;
case 1
    i=toc;
    %fprintf('%f..\n',i);
end

if i>(lval+skip-1)
    out=1;
    lval=i;
    fprintf('%d..',val);
    dcount=dcount+1;
    if (dcount>=feed);
        if (last>first)
          fprintf('%5.1f min, %5.1f min remain\n        ',toc/60,toc/60*(last-val+1)/(val-first));
        end
        dcount=0;
        out=2;
    end;
end
