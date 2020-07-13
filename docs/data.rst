.. _sepiadata:

SepiaData
=========

The main data container is `SepiaData`.
It should contain all simulation data and observed data (if applicable).
It also handles standardization and rescaling and creation of PCA and discrepancy bases (with interpolation to observed grid if needed).

.. autoclass:: sepia.SepiaData
    :members: