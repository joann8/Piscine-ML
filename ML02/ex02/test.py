import numpy as np
from loss import loss_

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([2, 14, -13, 5, 12, 4, -19,20])

print(loss_(X, Y))
print(loss_(X, X))
print(loss_(X, Z))