import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])

print("*********** TEST 1 ***************")
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
y_hat = mylr.predict_(X)
print("---> predict")
print(y_hat)
print("---> loss_elem_")
print(mylr.loss_elem_(Y, y_hat))
print("---> loss_")
print(mylr.loss_(Y,y_hat))

print("*********** TEST 2 ***************")
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print("---> theta") #pas pareil mais tout le reste de calculs OK --> erreur sujet
print(mylr.theta)
y_hat = mylr.predict_(X)
print("---> predict")
print(y_hat)
print("---> loss_elem_")
print(mylr.loss_elem_(Y,y_hat))
print("---> loss_")
print(mylr.loss_(Y,y_hat))
