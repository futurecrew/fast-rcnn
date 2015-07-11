import numpy as np

a = np.zeros((3, 4))

a[0, 3] = 1
a[1, 2] = 1
a[2, 1] = 1

print a

b = np.where(a == 1)

print b


c = np.arange(72)
c = c.reshape(2, 9, 4)

print c

c = c.reshape(2, 36)

print c
