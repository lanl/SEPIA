.. _aboutsepia:

About Sepia
===========

The general motivation for this project is the analysis of computer models, and the analysis of systems that have both
computer simulations and observed data. The underlying assumption is that experimental results are expensive or difficult
to collect, though it is possible to get simulation studies of the experimental problem in comparatively large, but still
restricted, quantities. Therefore a statistical model of system responses is to be constructed, which is called a surrogate
or emulator.

The emulator is constructed from an ensemble of simulation responses collected at various settings of input parameters.
The statistical model may be extended to incorporate system observations to improve system predictions and constrain,
or calibrate, unknown parameters. The statistical model accounts for the discrepancy between the system observations
and the simulation output. Parameters of the statistical model are determined through sampling their posterior
probability with MCMC. The statistical model can then be used in several ways. Statistical prediction of simulation and
observation response is relatively fast and includes uncertainty. Examination and analysis of the model parameters
can be used for calibration, screening, and analysis of model properties. The emulator may be further used for
sensitivity analysis, and other system diagnostics.

Before using the Sepia package, the analysis problem will have been defined, including: collecting system observations;
determining uncertain simulation parameters; establishing a design over the simulation parameters and running an
ensemble of simulations; considering an appropriate model for dimension reduction of the observation and simulation
response; considering an appropriate model for the discrepancy between simulation and observation; and considering
prior model parameter settings related to these issues.

The Sepia code is developed and maintained by the CCS-6 group at Los Alamos National Laboratory.

-----------

Approved by LANL/NNSA for software release: C19159 SEPIA

© 2020. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the
U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad
National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is
granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly,
and to permit others to do so.

This program is open source under the BSD-3 License. Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.