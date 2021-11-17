import numpy as np
import matplotlib.pyplot as plt

def predict_(x, theta):
    if isinstance(x,np.ndarray) and len(x) > 0 and \
        isinstance(theta,np.ndarray) and len(theta) == 2:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.reshape(x, (x.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        return np.matmul(X , theta)
    return None

def plot(x, y, theta):
    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x,np.ndarray) and len(x) > 0 and \
        isinstance(y,np.ndarray) and len(x) > 0 and \
        isinstance(theta,np.ndarray) and len(theta) == 2:
        plt.plot(x, y, 'o')
        X = predict_(x, theta)
        plt.plot(x, X)
        plt.show()

