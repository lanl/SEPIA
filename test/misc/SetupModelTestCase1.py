import unittest
import numpy as np
import scipy.io
import pickle

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel

# TODO currently broken due to reorganizing files, check later, may not need this test much anymore

np.random.seed(42)

class SetupModelTestCase1(unittest.TestCase):
    """
    Uses a canned test case and setup model results from GPMSA Matlab
    using script multi_sim_and_obs_sepia_gpmsa_test1.m (which loads the canned test case).
    So we compare GPMSA vs Sepia on setting up model given the same data.
    For now, just checking some precalculated likelihood stuff.
    """

    def setUp(self):

        # Create saved test case in python
        create_test_case()
        # Run SepiaModel in matlab GPMSA on saved test case
        # This currently doesn't work, just run matlab/multi_sim_and_obs_sepia_gpmsa_test1.m to be sure the .mat file is there
        #octave.generate_test('%s/matlab/' % os.path.dirname(os.path.abspath( __file__ )))

        with open('data/test_case_python_model.pkl', 'rb') as f:
            tmp = pickle.load(f)
            python_model = tmp['model']
        self.python_LamSim = python_model.num.LamSim
        self.python_SigObs = python_model.num.SigObs
        self.python_u = python_model.num.u
        self.python_v = python_model.num.v
        self.python_w = python_model.num.w

        matlab_model = scipy.io.loadmat('data/test_case_sepia_modelsetup.mat')
        self.matlab_LamSim = matlab_model['LamSim'].squeeze() # TODO check dims in python; matlab (3, 1), python (3, )
        self.matlab_SigObs = matlab_model['SigObs']
        self.matlab_u = matlab_model['u']
        self.matlab_v = matlab_model['v']
        self.matlab_w = matlab_model['w']


    def test_LamSim_matlabvpython(self):
        self.assertTrue(np.allclose(self.matlab_LamSim, self.python_LamSim))

    def test_SigObs_matlabvpython(self):
        self.assertTrue(np.allclose(self.matlab_SigObs, self.python_SigObs))

    def test_vuw_matlabvpython(self):
        self.assertTrue(np.allclose(self.matlab_u.reshape((-1, 1), order='F'), self.python_u))
        self.assertTrue(np.allclose(self.matlab_v.reshape((-1, 1), order='F'), self.python_v))
        self.assertTrue(np.allclose(self.matlab_w.reshape((-1, 1), order='F'), self.python_w))


def create_test_case():
    n_obs = 2
    n_sim = 5
    p = 3
    q = 4
    ell_sim = 80
    ell_obs = 20
    t = np.random.uniform(0, 1, (n_sim, q))
    x = 0.5 * np.ones((n_sim, p))
    y_ind = np.linspace(0, 100, ell_sim)
    y = 10 * np.random.normal(0, 1, (n_sim, 1)) * (y_ind[None, :] - 50)**2/75. + 20 * np.random.normal(0, 1, (n_sim, 1)) * y_ind[None, :] + 20 * np.random.normal(0, 1, (n_sim, 1))

    x_obs = 0.5 * np.ones((n_obs, p))
    y_obs_ind = np.linspace(10, 85, ell_obs)
    y_obs = 10 * np.random.normal(0, 1, (n_obs, 1)) * (y_obs_ind[None, :] - 50)**2/75. + 20 * np.random.normal(0, 1, (n_obs, 1)) * y_obs_ind[None, :] + 20 * np.random.normal(0, 1, (n_obs, 1))

    data = SepiaData(x_sim=x, t_sim=t, y_sim=y, y_ind_sim=y_ind, x_obs=x_obs, y_obs=y_obs, y_ind_obs=y_obs_ind)
    data.standardize_y()
    data.transform_xt()
    data.create_K_basis(n_pc=3)
    data.create_D_basis('constant')

    # Save as matfile for testing in matlab
    savedict = {'t':t, 'y':y, 'y_obs':y_obs, 'D':data.obs_data.D, 'Kobs':data.obs_data.K, 'Ksim':data.sim_data.K,
                'y_obs_std':data.obs_data.y_std, 'y_sim_std':data.sim_data.y_std, 'y_sd':data.sim_data.orig_y_sd}
    scipy.io.savemat('data/test_case_matlab.mat', savedict)

    g = SepiaModel(data)

    # Save pickle file of results
    savedict = {'model': g, 'data': data}
    with open('data/test_case_python_model.pkl', 'wb') as f:
        pickle.dump(savedict, f)




