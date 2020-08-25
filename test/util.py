

from functools import wraps
from time import time

def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time() - start
            print('Total execution time: %0.5g s' % (end_))
    return _time_it