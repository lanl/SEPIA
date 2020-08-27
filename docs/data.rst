.. _sepiadata:

SepiaData
=========

The main data container is `SepiaData`.
It should contain all simulation data and observed data (if applicable).
It also handles standardization and rescaling and creation of PCA and discrepancy bases (with interpolation to observed grid if needed).

The class `DataContainer` is used internally and generally not directly by users, but its attributes may be of interest.

.. autoclass:: sepia.SepiaData
    :members:
