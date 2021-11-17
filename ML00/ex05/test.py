import numpy as np
from plot import plot

x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

print("*********** TEST 1 ***************")
theta1 = np.array([4.5, -0.2])
plot(x, y, theta1)

print("*********** TEST 2 ***************")
theta2 = np.array([-1.5, 2])
plot(x, y, theta2)

print("*********** TEST 3 ***************")
theta3 = np.array([3, 0.3])
plot(x, y, theta3)

