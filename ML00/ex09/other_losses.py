import numpy as np
import math

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None


# FONCTIONS

def mse_elem(y, y_hat):
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        return (y_hat - y) **2
    return None

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
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        tmp = mse_elem(y, y_hat) * (1 / y.shape[0])
        return np.sum(tmp)
    return None

def rmse_elem(y, y_hat):
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        return (y_hat - y) **2
    return None

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
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        tmp = mse_elem(y, y_hat) * (1 / y.shape[0])
        return math.sqrt(np.sum(tmp))
    return None

def mae_elem(y, y_hat):
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        return abs(y_hat - y)
    return None

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
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        tmp = mae_elem(y, y_hat) * (1 / y.shape[0])
        return np.sum(tmp)
    return None

def r2score_elem(y, y_hat):
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        return (y - y.mean()) ** 2
    return None


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
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape and y.shape[1] == 1:
        top = mse_elem(y, y_hat)
        sum_top = np.sum(top)
        bottom = r2score_elem(y, y_hat)
        sum_bottom = np.sum(bottom)
        return 1 - (sum_top / sum_bottom)
    return None
