# SEPIA

Simulation-Enabled Prediction, Inference, and Analysis: physics-informed statistical learning.
This is a Python adaptation of [GPMSA](https://github.com/lanl/gpmsa).

<a href="https://doi.org/10.5281/zenodo.3979585"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3979585.svg" align="right" alt="DOI"></a>

<img src="docs/sepia.png" alt="sepia cuttlefish logo" width="150"/>

### What to Expect
SEPIA is intended to be a tool that enhances the collaboration between statisticians
and domain scientists who are using computational models to augment observations in
R&D and engineering applications. The code and the methodology 
it implements can be demonstrated simply, but new R&D often raises issues in 
analysis that are subtle and complicated. SEPIA has many options to address issues 
that have come up in the development team's experience in scientific applications,
and it is available to be extended to address new application requirements. We 
recommend the domain scientist consult or partner with a statistician familiar with the
methodology to ensure best outcomes. 

### Documentation
Current documentation is at [Read the Docs](http://sepia-lanl.readthedocs.io).
The documentation contains a workflow guide that is helpful for new users to read, and also contains a quick reference for basic commands as well as an API.

### Examples
Basic usage is demonstrated in the Examples directory. 
After looking at the documentation, check out the examples.

### Install package 
For cleaner package management and to avoid conflicts between different versions of packages,
we recommend installing inside an Anaconda or pip environment.
However, this is not required.

First, pull down the current source code from either by downloading a zip file or using `git clone`.

From the command line, while in the main SEPIA directory, use the following command to install sepia::

        pip install -e .[sepia]

The `-e` flag signals developer mode, meaning that if you update the code from Github, your installation will automatically
take those changes into account without requiring re-installation.
Some other essential packages used in SEPIA may be installed if they do not exist in your system or environment.


### Citing Sepia
Using Sepia in your work? Cite as:

James Gattiker, Natalie Klein, Earl Lawrence, & Grant Hutchings.
lanl/SEPIA. Zenodo. https://doi.org/10.5281/zenodo.3979584


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

