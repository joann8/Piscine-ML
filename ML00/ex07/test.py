import numpy as np
from vec_loss import loss_

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(loss_(X, Y))
print(loss_(X, X))

Z = np.empty((0,0))
print(loss_(X, Z))

Z = np.array([2, 14, -13, 5, 12, 4, -19, 20])
print(loss_(X, Z))

