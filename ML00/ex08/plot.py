import numpy as np
import matplotlib.pyplot as plt

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        if vect.shape[0] > 0 and vect.shape[1] == 1: 
            return vect
    return None

def check_theta(theta):
    if isinstance(theta,np.ndarray) and theta.shape[0] > 0:
        if np.ndim(theta) == 1:
            theta = np.reshape(theta, (theta.shape[0], 1))  
        if theta.shape[1] != 1:
            return None   
        return theta
    return None

def predict_(x, theta):
    x = check_dim_vector(x)
    theta = check_theta(theta)
    if x is not None and theta is not None:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.concatenate((ones, x), axis=1)
        return np.matmul(X , theta)
    return None

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    
    """
    x = check_dim_vector(x)
    y = check_dim_vector(y)
    theta = check_theta(theta)
    if x is not None and y is not None and theta is not None:
        plt.plot(x, y, 'o')
        X = predict_(x, theta)
        plt.plot(x, X)
        for i in range(0, len(x)):
            plt.vlines(x=x[i], ymin=min(y[i], X[i]), ymax=max(y[i], X[i]), color='red', linestyle='dashed')
        plt.show()