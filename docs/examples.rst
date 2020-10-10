.. _examples:

Examples
========

Current example code is located on GitHub: `Sepia examples`_.

.. _Sepia examples: https://github.com/lanl/SEPIA/tree/master/examples

A brief overview of helpful examples for new users is given here.

Example 1: synthetic data with univariate response
##################################################

This example creates toy synthetic data with a univariate response as a function of a single input.
It demonstrates how to create the `SepiaData` and `SepiaModel` objects before doing MCMC and visualizing the results.
Finally, emulator predictions and cross-validated emulator predictions are visualized using custom plotting code.
This is a good place to start to get used to the general Sepia workflow for simple problems.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Synthetic_toy_examples/univariate_example.ipynb>`_.

Example 2: synthetic data with multivariate response
####################################################

This example creates toy synthetic data with a multivariate response as a function of multiple inputs.
It demonstrates how to create the `SepiaData` and `SepiaModel` objects, including the creation of basis elements
for modeling the multivariate response.
It performs MCMC and shows some of the built-in visualization techniques in `SepiaPlot`.
Then, various kinds of predictions are made from the model and plotted.
This is a good place to start to get used to the general Sepia workflow for multivariate-output problems.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Synthetic_toy_examples/multivariate_example_with_prediction.ipynb>`_.

Example 3: tower example
########################

The tower examples involve a synthetic but more interpretable data set involving dropping balls from a tower and measuring
how long they take to reach certain heights from the ground.
The synthetic data is generated with a systematic discrepancy to the true generating process to demonstrate discrepancy modeling.
The main calibration parameter is an unknown drag coefficient, but we also investigate calibrating an additional parameter (coefficient of gravity).

* `Ball drop 1 <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Ball_Drop/ball_drop_1.ipynb>`_: simplest tower example with only one parameter calibrated
* `Ball drop 1 w/o discrepancy <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Ball_Drop/ball_drop_1_noD.ipynb>`_: similar to ball drop 1, but with no discrepancy in the model
* `Ball drop 2 <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Ball_Drop/ball_drop_2.ipynb>`_: similar to ball drop 1, but calibrates an additional parameter

Two variations on these examples demonstrate that Sepia accommodates ragged outputs -- that is, the observed multivariate data
can exist on different, non-aligned index sets.

* `Ball drop 1 with ragged observations <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Ball_Drop/ball_drop_1_ragged.ipynb>`_
* `Ball drop 2 with ragged observations <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Ball_Drop/ball_drop_2_ragged.ipynb>`_

Exmaple 4: Neddermeyer cylinder implosions
##########################################

The motivating experiment and synthetic data are discussed and analyzed in `Higdon et al (2008) <https://www.tandfonline.com/doi/abs/10.1198/016214507000000888>`_.
Briefly, this example uses synthetic data representing the time-varying radii of steel cylinders imploding due to high explosives.
Parameters to calibrate include the high explosive detonation energy and the yield stress of steel, with a controllable
experimental input (mass of high explosive).
This example is unique because the output is actually two-dimensional, requiring some custom setup for the basis functions
to represent the output -- so this is a good case study for custom basis setup in Sepia.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Neddermeyer/neddermeyer.ipynb>`_

Example 5: Al-5083 flyer plate
##############################

This example features real data that was analyzed previously via Bayesian model calibration in `Walters et al (2018) <https://aip.scitation.org/doi/abs/10.1063/1.5051442>`_.
The simulation design is over 11 inputs and the response is a time-indexed velocity of the Al-5083 aluminum alloy after impact from which
important feature points have been extracted for each curve.
The pairs plot of the calibration parameter posterior distributions is consistent with the 2018 paper.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Al_5083/Al_5083_calibration.ipynb>`_

Other examples
##############

The remaining examples show more advanced or customized usages of Sepia.

* Categorical input variables: `link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Synthetic_toy_examples/univariate_example_categorical_variable.ipynb>`_
* Neddermeyer with separate groups of parameters for different discrepancy basis elements: `link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Neddermeyer/neddermeyer_lamVzGroup.ipynb>`_
* Neddermeyer used to demo shared and hierarchical theta models: `link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Neddermeyer/neddermeyer_shared_hierarchical.ipynb>`_
* Tower with parallel chain sampling (not a notebook): `link <http://www.github.com/lanl/SEPIA/tree/master/examples/Ball_Drop/ball_drop_1_parallelchains.py>`_
