.. _model-math:

SEPIA model math
================

There are :math:`n` physical (observational) experiments.
From the :math:`i`th physical experiment at :math:`p` inputs :math:`{\bf x}^{obs}_i=(x^{obs}_{i1}, \ldots, x^{obs}_{ip})`,
the observation :math:`{\bf y}^{obs}_{i}({\bf x}^{obs}_i)` (an :math:`n_{y^{obs}_{i}} \times 1` vector) is modeled by

.. math:: {\bf y}^{obs}_{i}({\bf x}^{obs}_{i})= \boldsymbol \eta({\bf x}^{obs}_{i},\boldsymbol \theta)+ \boldsymbol \delta({\bf x}^{obs}_{i}) + {\bf e}^{obs}_{i},

where the observation error vector :math:`{\bf e}^{obs}_{i}` is modeled by

.. math:: {\bf e}^{obs}_{i} \sim MVN\left({\bf 0}_{n_{y^{obs}_{i}}}, \, \frac{1}{\lambda_{y^{obs}}^{\tt Os}} \Sigma^{obs}_i \right).

:math:`\boldsymbol \eta(\cdot)` is an emulator from a simulation code, :math:`\boldsymbol \theta` corresponds to inputs of the parameter,
and :math:`\boldsymbol \delta(\cdot)` is a discrepancy from reality.

There are :math:`m` simulation experiments.
From the :math:`i`th simulation experiment at :math:`p+q` inputs :math:`{\bf x}^{sim}_i=(x^{sim}_{i1}, \ldots, x^{sim}_{ip})`
and :math:`{\bf t}^{sim}_i=(t^{sim}_{i1}, \ldots, t^{sim}_{iq})`, the observation :math:`{\bf y}^{sim}_{i}({\bf x}^{sim}_i,{\bf t}^{sim}_i)`
(an :math:`n_{y^{sim}_{i}} \times 1`$` vector) is modeled by

.. math:: {\bf y}^{sim}_{i}({\bf x}^{sim}_i,{\bf t}^{sim}_i)= \boldsymbol \eta({\bf x}^{sim}_i,{\bf t}^{sim}_i)+ {\bf e}^{sim}_{i},

where the error vector :math:`{\bf e}^{sim}_{i}` is modeled by :math:`MVN\left({\bf 0}_{n_{y^{sim}_{i}}}, \, \frac{1}{\lambda^{\tt WOs}_{y^{sim}}} {\bf I} \right)`
and :math:`I_m` is the :math:`m \times m` identity matrix.

We re-express :math:`\boldsymbol \eta({\bf x}_i,{\bf t}_i)` and :math:`\boldsymbol \delta({\bf x}_i)` by linear combinations of
basis functions and approximate them using a subset of the complete set of basis functions.
Consequently,

.. math:: \boldsymbol \eta({\bf x}^{obs}_i,\boldsymbol \theta) \approx \sum_{j=1}^{p_u} {\bf K}^{obs}_j u_j({\bf x}^{obs}_i, \boldsymbol \theta)

for :math:`p_{u}` basis functions :math:`{\bf K}^{obs}_j`.
So the matrix :math:`{\bf K}^{obs}=({\bf K}^{obs}_1 \cdots {\bf K}^{obs}_{p_u})`.
Similarly,

.. math:: \boldsymbol \delta({\bf x}^{obs}_i) \approx \sum_{j=1}^{p_v} {\bf D}^{obs}_j v_j({\bf x}^{obs}_i)

for :math:`p_{v}` basis functions :math:`{\bf D}^{obs}_j`.
So the matrix :math:`{\bf D}^{obs}=({\bf D}^{obs}_1 \cdots {\bf D}^{obs}_{p_v})`.

For the simulations,

.. math:: \boldsymbol \eta({\bf x}^{sim}_i,{\bf t}^{sim}_i) \approx \sum_{j=1}^{p_u} {\bf K}^{sim}_j w_j({\bf x}^{sim}_i,{\bf t}^{sim}_i)

for :math:`p_{u}` basis functions :math:`{\bf K}^{sim}_j`, where :math:`w_j({\bf x}^{sim}_i,{\bf t}^{sim}_i)=u_j({\bf x}^{sim}_i,{\bf t}^{sim}_i)+\epsilon^{sim,nug}_j`.
So the matrix :math:`{\bf K}^{sim}=({\bf K}^{sim}_1 \cdots {\bf K}^{sim}_{p_u})`.

Note that :math:`\frac{1}{\lambda^{\tt Ws}_{\epsilon^{sim,nug}_j}}` is the variance of an i.i.d. Normally distributed nugget
:math:`\epsilon^{sim,nug}_j` with mean 0 to account for small numerical fluctuations in the simulator.
The nugget is used only in fitting, not in prediction.

In the above equations, any error from the truncated basis approximations is assumed to be part of :math:`{\bf e}^{obs}_{i}` or :math:`{\bf e}^{sim}_{i}`.

The :math:`u_j({\bf x},{\bf t}), \, j=1, \ldots, p_u` are modeled as a GP with mean :math:`{\bf 0}_{n}` and variance covariance matrix :math:`\frac{1}{\lambda^{\tt Uz}_{u_j}}R^{u}_j`, where

.. math:: R^{u}_j(({\bf x}_i,{\bf t}_i)),({\bf x}_l,{\bf t}_l))=\prod_{k=1}^p \left({\rho^{u}_{jk}}\right)^{4|x_{ik}-x_{lk}|^2} \prod_{k=1}^q \left({\rho^{u}_{(j+p)k}}\right)^{4 |t_{ik}-t_{lk}|^2}.

A more familiar form (revealing the squared exponential covariance function form) is

.. math:: R^{u}_j(({\bf x}_i,{\bf t}_i)),({\bf x}_l,{\bf t}_l))=\prod_{k=1}^p \exp(-{{\beta^{u}_{jk}}}|x_{ik}-x_{lk}|^2) \prod_{k=1}^q \exp(-{{\beta^{u}_{(j+p)k}}}|x_{ik}-x_{lk}|^2),

so that :math:`\beta^{u}_{jk}= -{4}\log\left(\rho^u_{jk}\right)` or :math:`\rho^u_{jk}=\exp\left(-\frac{\beta^u_{jk}}{4}\right)`.

Similarly, the :math:`v_j({\bf x}^{obs}_i), \, j=1, \ldots, n` are modeled as a GP with mean :math:`{\bf 0}_{n}` and
variance covariance matrix :math:`\frac{1}{\lambda^{\tt Vz}_{v^{obs}_j}}R^{v}_j`, where

.. math:: R^{v}_j({\bf x}^{obs}_i,{\bf x}^{obs}_l)=\prod_{k=1}^p ({\rho^{v}_{jk}})^{4|x_{ik}-x_{lk}|^2},

whose more familiar form is

.. math:: R^{v}_j({\bf x}^{obs}_i,{\bf x}^{obs}_l)=\prod_{k=1}^q \exp(-{{\beta^{v}_{jk}}}|x_{ik}-x_{lk}|^2),

so that :math:`\beta^{v}_{jk}= -4\log\left(\rho^v_{jk}\right)` or :math:`\rho^v_{jk}=\exp\left(-\frac{\beta^v_{jk}}{4}\right)`.

Note that the :math:`{\bf x}, {\bf t}, \boldsymbol \theta` are transformed to [0, 1] and
the :math:`{\bf y}^{obs}_{i}({\bf x}^{obs}_{i})` and :math:`{\bf y}^{sim}_{i}({\bf x}^{sim}_i,{\bf t}^{sim}_i)` are normalized
to have sample mean :math:`{\bf 0}` and covariance matrices equal to identity matrices.
Consequently, the :math:`\Sigma^{obs}_i` for :math:`{\bf y}^{obs}_{i}({\bf x}^{obs}_{i})` has to be normalized in the same way that the :math:`{\bf y}^{obs}_{i}({\bf x}^{obs}_{i})` are.


