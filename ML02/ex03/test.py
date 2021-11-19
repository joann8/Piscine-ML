import numpy as np
from gradient import gradient

x = np.array([[ -6, -7, -9], [ 13, -2, 14], [ -7, 14, -1], [ -8, -4, 6], [ -5, -9, 6], [ 1, -5, 11], [ 9, -11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19])

theta1 = np.array([0, 3, 0.5, -6])
print(gradient(x, y, theta1))

theta2 = np.array([0, 0, 0, 0])
print(gradient(x, y, theta2))
