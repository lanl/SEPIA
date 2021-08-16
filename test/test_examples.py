import sys
sys.path.append('../examples/Neddermeyer/')
from neddermeyer import neddermeyer_example
#sys.path.append('../examples/Ball_Drop/')
#from ball_drop_1 import ball_drop_1_example
#from ball_drop_2 import ball_drop_2_example

import unittest
import numpy as np
np.random.seed(42)

#class SepiaBallDropTestCase(unittest.TestCase):
#   ball_drop_2_example(test=1,datadir='/../examples/Ball_Drop/data/data_ball_drop_2/')
#   ball_drop_1_example(test=1)
    
class SepiaNeddermeyerTestCase(unittest.TestCase):
    neddermeyer_example(test=1)
