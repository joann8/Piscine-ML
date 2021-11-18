from numpy import np
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

def plot(lnr, x, y):
    if isinstance(lnr, MyLR):
        if isinstance(x, np.ndarray) and len(x) > 0 and \
            isinstance(y, np.ndarray) and len(x) > 0 :
            plt.plot(x, y, 'o')
            X = lnr.predict_(x, lnr.theta)
            plt.plot(x, X)
            plt.show()