import numpy as np
M = np.arange(2, 27)
print(M)

M = M.reshape(5, 5)
print(M)

M[1:-1, 1:-1] = 0
print(M)

M = M.dot(M)
print(M)

v = M[0]
magnitude_v = np.sqrt(np.sum(v**2))
print(magnitude_v)