.. _sepiapredict:

SepiaPredict
============

The `SepiaPredict` class is used to obtain predictions from the model based on MCMC samples.
Three types of predictions are possible:
    - SepiaEmulatorPrediction: predictions from the emulator portion of the model
    - SepiaXvalEmulatorPrediction: cross-validated predictions from the emulator
    - SepiaFullPrediction: predictions from the full model (including discrepancy and observation noise)

The `SepiaPredict` classes handle transformation from `u`/`w`/`v` to the original output space.

.. autoclass:: sepia.SepiaPrediction
    :members:

.. autoclass:: sepia.SepiaXvalEmulatorPrediction
    :members:

.. autoclass:: sepia.SepiaEmulatorPrediction
    :members:

.. autoclass:: sepia.SepiaFullPrediction
    :members:


