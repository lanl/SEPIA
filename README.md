# SEPIA

Simulation-Enabled Prediction, Inference, and Analysis: physics-informed statistical learning.
This will be the python version of [GPMSA](https://github.com/lanl/gpmsa)

### Warning: still under development!
Basic functionality (model setup, likelihood evaluation, and mcmc sampling) is complete and has been tested against GPMSA matlab.
Some features are untested or still being developed (like auto step size tuning and sensitivity analysis).

### Examples
Basic usage is demonstrated in the Examples directory. 
Follow one of these examples but plugging in your own data.

### Install package locally
After cloning git repo, use the following command from this directory to install locally.

`pip install -e .[sepia]`

 I recommend doing this inside a python conda environment.
 The packages installed in the development environment are listed in `environment.yml`.
Use `conda env create -f environment.yml` to create the environment, then activate as `source activate sepia` before installing sepia.

### Documentation
You can build it locally if you wish following docs/README.md, or view version on
[Read the Docs](http://sepia-lanl.readthedocs.io).

---

Approved by LANL/NNSA for software release: C19159 SEPIA 

© 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution. 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

