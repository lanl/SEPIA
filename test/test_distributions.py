import unittest
import numpy as np
import sepia.distributions as dist

np.random.seed(0)

class TestDistributions(unittest.TestCase):
    def test_beta(self):
        X = dist.Beta(3, 2)
        lpdf = X.logpdf(0.6)

        # Test logpdf.
        self.assertAlmostEqual(lpdf, 0.54696, 4)

        # Test transformation and inv_transform.

        # Test log jac.

        # Test support
