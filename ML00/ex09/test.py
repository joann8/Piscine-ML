import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from other_losses import mse_, rmse_, mae_, r2score_
from math import sqrt

x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])

print("*********** MSE ***************")
print(mse_(x,y))
print(mean_squared_error(x,y))

print("*********** RMSE ***************")
print(rmse_(x,y))
print(sqrt(mean_squared_error(x,y)))

print("*********** MAE ***************")
print(mae_(x,y))
print(print(mean_absolute_error(x,y)))

print("*********** R2SCORE***************")
print(r2score_(x,y))
print(r2_score(x,y))
