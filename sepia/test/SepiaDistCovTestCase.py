import unittest
import numpy as np

from sepia.SepiaDistCov import SepiaDistCov

np.random.seed(42)

class SepiaDistCovTestCase(unittest.TestCase):


    def setUp(self):
        n1 = 100
        n2 = 20
        self.lams = 10.
        self.lamz = 1.
        # Unidimensional
        self.beta1 = np.array([np.exp(-0.25 * 0.2)])[:, None]
        self.x1 = np.linspace(0, 1, n1)[:, None]
        self.x2 = np.linspace(0, 1, n2)[:, None]
        # Multi dimensional
        self.beta2 = np.exp(-0.25 * np.linspace(0, 1, 5))[:, None]
        self.X1 = np.random.uniform(0, 1, (n1, 5))
        self.X2 = np.random.uniform(0, 1, (n2, 5))


    def test_square_dist_unidim_x(self):
        n = self.x1.shape[0]
        sqdistmanual = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sqdistmanual[i, j] = np.square(self.x1[i] - self.x1[j])
        cov = SepiaDistCov(self.x1)
        sqdistmat = np.zeros((cov.n, cov.n))
        sqdistmat[cov.ind] = cov.sqdist[:, 0]
        sqdistmat = sqdistmat + sqdistmat.T
        self.assertTrue(np.allclose(sqdistmanual, sqdistmat))


    def test_square_dist_multidim_x(self):
        n = self.X1.shape[0]
        sqdistmanual = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sqdistmanual[i, j] = np.sum(np.square(self.X1[i, :] - self.X1[j, :]))
        cov = SepiaDistCov(self.X1)
        sqdistmat = np.zeros((cov.n, cov.n))
        sqdistmat[cov.ind] = np.sum(cov.sqdist, axis=-1)
        sqdistmat = sqdistmat + sqdistmat.T
        self.assertTrue(np.allclose(sqdistmanual, sqdistmat))


    def test_rect_dist_unidim_x(self):
        m, n = self.x1.shape[0], self.x2.shape[0]
        sqdistmanual = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                sqdistmanual[i, j] = np.square(self.x1[i] - self.x2[j])
        cov = SepiaDistCov(self.x1, self.x2)
        sqdistmat = np.zeros((cov.n, cov.m))
        sqdistmat[cov.ind] = np.sum(cov.sqdist, axis=-1)
        self.assertTrue(np.allclose(sqdistmanual, sqdistmat))


    def test_rect_dist_multidim_x(self):
        m, n = self.X1.shape[0], self.X2.shape[0]
        sqdistmanual = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                sqdistmanual[i, j] = np.sum(np.square(self.X1[i, :] - self.X2[j, :]))
        cov = SepiaDistCov(self.X1, self.X2)
        sqdistmat = np.zeros((cov.n, cov.m))
        sqdistmat[cov.ind] = np.sum(cov.sqdist, axis=-1)
        self.assertTrue(np.allclose(sqdistmanual, sqdistmat))


    def test_square_cov_unidim_x(self):
        n = self.x1.shape[0]
        covmanual = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                nug = 0
                if i == j:
                    nug = 1/self.lams
                covmanual[i, j] = np.exp(-np.square(self.x1[i] - self.x1[j]) * np.squeeze(self.beta1))/self.lamz + nug
        cov = SepiaDistCov(self.x1).compute_cov_mat(self.beta1, self.lamz, self.lams)
        self.assertTrue(np.allclose(covmanual, cov))


    def test_square_cov_multidim_x(self):
        n = self.X1.shape[0]
        covmanual = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                nug = 0
                if i == j:
                    nug = 1/self.lams
                covmanual[i, j] = np.exp(-np.sum(np.square(self.X1[i] - self.X1[j]) * np.squeeze(self.beta2), axis=-1))/self.lamz + nug
        cov = SepiaDistCov(self.X1).compute_cov_mat(self.beta2, self.lamz, self.lams)
        self.assertTrue(np.allclose(covmanual, cov))


    def test_rect_cov_unidim_x(self):
        m, n = self.x1.shape[0], self.x2.shape[0]
        covmanual = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                covmanual[i, j] = np.exp(-np.square(self.x1[i] - self.x2[j]) * np.squeeze(self.beta1))/self.lamz
        cov = SepiaDistCov(self.x1, self.x2).compute_cov_mat(self.beta1, self.lamz)
        self.assertTrue(np.allclose(covmanual, cov))


    def test_rect_cov_multidim_x(self):
        m, n = self.X1.shape[0], self.X2.shape[0]
        covmanual = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                covmanual[i, j] = np.exp(-np.sum(np.square(self.X1[i] - self.X2[j]) * np.squeeze(self.beta2), axis=-1))/self.lamz
        cov = SepiaDistCov(self.X1, self.X2).compute_cov_mat(self.beta2, self.lamz)
        self.assertTrue(np.allclose(covmanual, cov))




