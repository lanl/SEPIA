# NOTE:
# date: 20 Aug, 2021.
# These tests are currently failing. Issues will be addressed later.
# See: Issue #33.

# import sys
# from examples.run_nb import run_notebook
# import unittest
# import numpy as np
# np.random.seed(42)
#
# class SepiaBallDropTestCase(unittest.TestCase):
#     def test_ball_drop_1(self):
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_1.ipynb',
#                         execute_path='examples/Ball_Drop/',html=0)
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_1_noD.ipynb',
#                 execute_path='examples/Ball_Drop/',html=0)
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_1_ragged.ipynb',
#                 execute_path='examples/Ball_Drop/',html=0)
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_1_parallelchains.ipynb',
#                 execute_path='examples/Ball_Drop/',html=0)
#     def test_ball_drop_2(self):
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_2.ipynb',
#                     execute_path='examples/Ball_Drop/',html=0)
#         run_notebook(notebook_filename='examples/Ball_Drop/ball_drop_2_ragged.ipynb',
#                     execute_path='examples/Ball_Drop/',html=0)
#
# class SepiaNeddermeyerTestCase(unittest.TestCase):
#     def test_neddermeyer(self):
#         run_notebook(notebook_filename='examples/Neddermeyer/neddermeyer.ipynb',
#                     execute_path='examples/Neddermeyer/',html=0)
#         run_notebook(notebook_filename='examples/Neddermeyer/neddermeyer_shared_hierarchical.ipynb',
#                     execute_path='examples/Neddermeyer/',html=0)
#         run_notebook(notebook_filename='examples/Neddermeyer/neddermeyer_lamVzGroup.ipynb',
#                     execute_path='examples/Neddermeyer/',html=0)
