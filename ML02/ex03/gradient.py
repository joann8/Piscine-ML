import numpy as np

def check_dim_maxtrix(mat):
    if isinstance(mat, np.ndarray):
        if np.ndim(mat) == 1:
            mat = np.reshape(mat, (mat.shape[0], 1))       
        if mat.shape[0] > 0 and mat.shape[1] > 0:
            return mat
    return None

def check_theta(theta, n):
    if isinstance(theta,np.ndarray) and theta.shape[0] > 0:
        if np.ndim(theta) == 1:
            theta = np.reshape(theta, (theta.shape[0], 1))  
        if theta.shape[0] == n + 1 and theta.shape[1] == 1:  
            return theta
    return None

def add_intercept(x):
    if isinstance(x,np.ndarray) and len(x) > 0:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        return np.concatenate((ones, x), axis=1)    
    return None

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible shapes.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        x = check_dim_maxtrix(x)
        y = check_dim_maxtrix(y)
        if x is not None and y is not None and x.shape[0] == y.shape[0]:
            theta = check_theta(theta, x.shape[1])
            if theta is not None:
                X = add_intercept(x)
                Xtrans = np.transpose(X)
                X_th_y = np.matmul(X, theta) - y
                return ((1 / x.shape[0]) * np.matmul(Xtrans, X_th_y)) # format output? a revoir  
        return None
    except Exception as err:
        print(err)
        return None