import unittest
import numpy as np
from sepia.DataContainer import DataContainer

class DataContainerTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 10
        # Scalar output, one-dimensional x/t
        self.dc1 = DataContainer(x=np.linspace(-1, 2, self.n), y=np.linspace(-3, 5, self.n),
                                 t=np.linspace(-5, 5, self.n))
        # Scalar output, one-dimensional x/t but with second dimension equal to 1
        self.dc2 = DataContainer(x=np.random.uniform(-1, 2, (self.n, 1)), y=np.random.uniform(-3, 5, (self.n, 1)),
                                 t=np.random.uniform(-5, 5, (self.n, 1)))
        # Scalar output, multi-dimensional x/t
        self.dc3 = DataContainer(x=np.random.uniform(-1, 2, (self.n, 3)), y=np.random.uniform(-3, 5, (self.n, 1)),
                                 t=np.random.uniform(-5, 5, (self.n, 5)))
        # Multi-output, multi-dimensional x/t
        self.dc4 = DataContainer(x=np.random.uniform(-1, 2, (self.n, 3)), y=np.random.uniform(-3, 5, (self.n, 50)),
                                 t=np.random.uniform(-5, 5, (self.n, 5)), y_ind=np.linspace(0, 10, 50))
        # Multi-dimensional x/t with some columns taking identical values
        self.dc5 = DataContainer(x=np.concatenate([0.5*np.ones((self.n, 1)), np.random.uniform(-1, 2, (self.n, 2))], axis=1),
                                 y=np.random.uniform(-3, 5, (self.n, 50)), y_ind=np.linspace(0, 10, 50),
                                 t=np.concatenate([0.5*np.ones((self.n, 1)), np.random.uniform(-5, 5, (self.n, 4))], axis=1))


    def test_xy_size(self):
        # Make sure it promotes 1D vectors to (n, 1)
        self.assertEqual(self.dc1.x.shape, (self.n, 1), 'incorrect x size')
        self.assertEqual(self.dc1.y.shape, (self.n, 1), 'incorrect y size')
        self.assertEqual(self.dc1.t.shape, (self.n, 1), 'incorrect t size')
        # Make sure still works if given (n, 1)
        self.assertEqual(self.dc2.x.shape, (self.n, 1), 'incorrect x size')
        self.assertEqual(self.dc2.y.shape, (self.n, 1), 'incorrect y size')
        self.assertEqual(self.dc2.t.shape, (self.n, 1), 'incorrect t size')
        # Check if given t (n, q) and x (n, p)
        self.assertEqual(self.dc3.x.shape, (self.n, 3), 'incorrect x size')
        self.assertEqual(self.dc3.y.shape, (self.n, 1), 'incorrect y size')
        self.assertEqual(self.dc3.t.shape, (self.n, 5), 'incorrect t size')
        # Check if given t (n, q) and x (n, p) and multi-output y
        self.assertEqual(self.dc4.x.shape, (self.n, 3), 'incorrect x size')
        self.assertEqual(self.dc4.y.shape, (self.n, 50), 'incorrect y size')
        self.assertEqual(self.dc4.t.shape, (self.n, 5), 'incorrect t size')
        self.assertEqual(self.dc4.y_ind.shape[0], self.dc4.y.shape[1], 'y/y_ind shape mismatch')


    #def test_compute_PCA_basis(self):
