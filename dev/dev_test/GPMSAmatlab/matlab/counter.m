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
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
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
    skip=arg4+1;  feed=arg5;
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
