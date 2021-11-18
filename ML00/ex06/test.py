import numpy as np
from loss import loss_
from loss import loss_elem_
from loss import predict_

print("*********** TEST 1 ***************")
x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(loss_elem_(y1, y_hat1))
print(loss_(y1, y_hat1))

print("*********** TEST 2 ***************")
x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])
print(loss_elem_(y2, y_hat2))
print(loss_(y2, y_hat2))

print("*********** TEST 3 ***************")
x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])
print(loss_(y3, y_hat3))
print(loss_(y3, y3))