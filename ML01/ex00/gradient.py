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

def predict_(x, theta):
    x = check_dim_vector(x)
    theta = check_theta(theta)
    if x is not None and theta is not None:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.concatenate((ones, x), axis=1)
        return np.matmul(X , theta)
    return None

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    x = check_dim_vector(x)
    y = check_dim_vector(y)
    theta = check_theta(theta)
    if x is not None and y is not None and theta is not None \
        and x.shape == y.shape and y.shape[1] == 1:
        elem = predict_(x, theta) - y
        gradient0 = np.sum(elem ) / x.shape[0]
        gradient1 = np.sum((elem * x)) / x.shape[0]
        return np.array([gradient0, gradient1])        
    return None
