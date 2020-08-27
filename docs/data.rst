.. _sepiadata:

SepiaData
=========

The main data container is `SepiaData`.
It should contain all simulation data and observed data (if applicable).
It also handles standardization and rescaling and creation of PCA and discrepancy bases (with interpolation to observed grid if needed).

The `DataContainer` class is used by `SepiaData` and not usually directly by users, but some of its attributes may be useful to access.

.. autoclass:: sepia.SepiaData
    :members:

.. autoclass:: sepia.DataContainer
    :members:
