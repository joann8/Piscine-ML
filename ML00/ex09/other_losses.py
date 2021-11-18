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
    
def mse_elem(y, y_hat):
def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """

def rmse_elem(y, y_hat):
def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """

def mae_elem(y, y_hat):
def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """

def r2score_elem(y, y_hat):
def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """
