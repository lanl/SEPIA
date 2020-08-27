# Test appending to list vs preallocating

import numpy as np
from test.util import timeit


class listContainer:
    def __init__(self):
        self.res = []


class arrContainer:
    def __init__(self, arr_shape):
        self.res = np.empty(arr_shape)


arr = np.random.normal(size=(5, 1))

@timeit
def test_append(n, arr):
    lc = listContainer()
    for i in range(n):
        lc.res.append(arr)

@timeit
def test_prealloc(n, arr):
    ac = arrContainer((*arr.shape, n))
    for i in range(n):
        ac.res[:, :, i] = arr

for pow in np.arange(2, 8):
    print('\n 10^%d values' % pow)
    print('append to list')
    test_append(np.power(10, pow), arr)
    print('prealloc np array')
    test_prealloc(np.power(10, pow), arr)

