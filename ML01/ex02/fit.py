import numpy as np

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None

def check_theta(theta):
    if isinstance(theta,np.ndarray) and theta.shape[0] > 0:
        if np.ndim(theta) == 1:
            theta = np.reshape(theta, (theta.shape[0], 1))  
        if theta.shape[0] == 2 and theta.shape[1] == 1:
            return theta 
    return None

def check_alpha(alpha):
    if isinstance(alpha, float) and alpha >= 0 and alpha <= 1:
        return alpha
    return None  

def predict(x, theta):
    x = check_dim_vector(x)
    theta = check_theta(theta)
    if x is not None and theta is not None:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.concatenate((ones, x), axis=1)
        return np.matmul(X , theta)
    return None

def gradient(x, y, theta):
    x = check_dim_vector(x)
    y = check_dim_vector(y)
    theta = check_theta(theta)
    if x is not None and y is not None and theta is not None \
        and x.shape == y.shape and y.shape[1] == 1:
        ones = np.ones((x.shape[0], 1)) 
        X = np.concatenate((ones, x), axis=1)
        X_th_y = np.matmul(X , theta) - y
        Xtrans= np.matrix.transpose(X)
        return ((1 / x.shape[0]) * np.matmul(Xtrans, X_th_y))
    return None

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    x = check_dim_vector(x)
    y = check_dim_vector(y)
    theta = check_theta(theta)
    alpha = check_alpha(alpha)
  
    if x is not None and y is not None and theta is not None and alpha is not None\
        and x.shape == y.shape and y.shape[1] == 1 \
        and isinstance(max_iter, int) and max_iter > 0:
        for i in range(max_iter):
            theta = theta - alpha * gradient(x, y, theta)
        return theta   
    return None