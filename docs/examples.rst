.. _examples:

Examples
========

Current example code is located on GitHub: `Sepia examples`_.

.. _Sepia examples: https://github.com/lanl/SEPIA/tree/master/examples

A brief overview of helpful examples for new users is given here.

Example 1: synthetic data with univariate response
##################################################

This example creates synthetic data with a univariate response as a function of a single input.
It demonstrates how to create the `SepiaData` object and `SepiaModel` object before doing MCMC and visualizing the results.
Finally, emulator predictions and cross-validated emulator predictions are visualized using custom plotting code.
This is a good place to start to get used to the general Sepia workflow.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Synthetic_toy_examples/univariate_example.ipynb>`_.


Example 2: synthetic data with multivariate response
####################################################

This example creates synthetic data with a multivariate response as a function of multiple inputs.
It demonstrates how to create the `SepiaData` object and `SepiaModel` objects, including the creation of basis elements
for modeling the multivariate response.
It performs MCMC and shows some of the built-in visualization techniques in `SepiaPlot`.
Then, various kinds of predictions are made from the model and plotted.
This is a good place to start if you have a problem with multivariate responses.

`Notebook link <https://nbviewer.jupyter.org/github/lanl/SEPIA/blob/master/examples/Synthetic_toy_examples/multivariate_example_with_prediction.ipynb>`_.
