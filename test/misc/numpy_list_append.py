import time
import numpy as np

N = 1e5

# using list -> array
start = time.time()
result = []
for i in range(int(N)):
    result.append(i)
result = np.array(result)
end = time.time()
print("Elapsed time using list = %0.3g sec" % ((end - start)))

# using np.append
start = time.time()
result = np.array(0)
for i in range(1, int(N)):
    result = np.append(result, i)
end = time.time()
print("Elapsed time using np.append = %0.3g sec" % ((end - start)))