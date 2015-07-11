import numpy as np

a = np.zeros((10, 4))

b = [2, 5]
c = [[1, 2, 3, 4], [5, 6, 7, 8]]

print a

a[b] = c

print a

d = np.swapaxes(a, 0, 1)

print d
print d.shape