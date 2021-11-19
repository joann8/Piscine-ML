import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

def plot_linear(x, y, y_hat):
    try:
        plt.scatter(x, y, marker='o', color='cyan', label='$S_{true}(pills)$')
        plt.scatter(x, y_hat, marker='s', color='springgreen', label='$S_{predic}(pills)$)')
        plt.plot(x, y_hat, '--',color='springgreen')
        plt.xlabel("Quantity of blue pill (in micrograms)")
        plt.ylabel("Space driving score")
        plt.legend()
        plt.show()
    except Exception as err:
        print(err)
        pass

#def plot_loss(x, y, theta):
    # A FAIRE

data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

print("*********** TEST 1 ***************")
linear_model1 = MyLR(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict_(Xpill)
print(linear_model1.mse_(Yscore, Y_model1))
print(mean_squared_error(Yscore, Y_model1))

plot_linear(Xpill, Yscore, Y_model1)
#plot_loss(Xpill, Yscore, linear_model1.theta)



print("*********** TEST 2 ***************")
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model2 = linear_model2.predict_(Xpill)
print(linear_model2.mse_(Yscore, Y_model2))
print(mean_squared_error(Yscore, Y_model2))

plot_linear(Xpill, Yscore, Y_model2)
#plot_loss(Xpill, Yscore, linear_model2.theta)




