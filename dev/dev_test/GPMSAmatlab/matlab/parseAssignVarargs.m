% function parseAssignVarargs(validVars)
% assigns specified caller varargs to the corresponding variable name
%   in the calling workspace. vars not specified are not assigned.
% validVars is a cell array of strings that represents possible
%    arg names, and the variable name in the workspace (identical)
% varargs is the varargin cell array to parse

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James R. Gattiker, Los Alamos National Laboratory
% © 2008. Triad National Security, LLC. All rights reserved.
% This file was distributed as part of the GPM/SA software package
% Los Alamos Computer Code release LA-CC-06-079, C-06,114
% github.com/lanl/GPMSA
% Full copyright in the README.md in the repository
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function parseAssignVarargs(validVars)

pr=0;

bargs=evalin('caller','varargin');

for ii=1:2:length(bargs)
  varID=find(strcmp(validVars,bargs{ii}));
  if length(varID)~=1;
    ME=MException('VerifyInput:InvalidArgument', ...
                  sprintf('invalid optional argument key: %s',bargs{ii}) );
    throwAsCaller(ME);
  end
  if pr; fprintf('Assigning: %s\n', validVars{varID}); bargs{ii+1}, end 
  if evalin('caller',['exist(''' validVars{varID} ''');'])
     assignin('caller',validVars{varID},bargs{ii+1});
  else
    error('Variable (argument) %s not initialized\n',validVars{varID});
  end
  
end
 