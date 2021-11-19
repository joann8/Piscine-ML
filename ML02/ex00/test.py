import numpy as np
from prediction import simple_predict

x = np.arange(1,13).reshape((4,-1))

theta1 = np.array([5, 0, 0, 0])
print(simple_predict(x, theta1))

theta2 = np.array([0, 1, 0, 0])
print(simple_predict(x, theta2))

theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
print(simple_predict(x, theta3))

theta4 = np.array([-3, 1, 2, 3.5])
print(simple_predict(x, theta4))
