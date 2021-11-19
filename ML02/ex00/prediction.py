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

def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        x = check_dim_maxtrix(x)
        if x is not None:
            theta = check_theta(theta, x.shape[1])   
            if theta is not None: 
                return np.matmul(add_intercept(x), theta) # format OK en colonne?
        return None
    except Exception as err:
        print(err)
        return None