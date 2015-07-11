import numpy as np

a = np.zeros((3, 4))

print a

a[0, :] = [1, 2, 3, 4]
a[1, :] = [5, 6, 7, 8]
a[2, :] = [9, 10, 11, 12]

print a

b = a[:, (1, 0, 2, 3)]

print b

c = np.swapaxes(a, 0, 1)

print c
