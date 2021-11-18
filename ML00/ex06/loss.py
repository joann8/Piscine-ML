import numpy as np

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None

def check_theta(theta):
    if isinstance(theta,np.ndarray) and theta.shape[0] > 0 and theta.shape[1] == 1:       
        return 0
    return 1

def predict_(x, theta):
    x = check_dim_vector(x)
    if x is not None and check_theta(theta) == 0:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.concatenate((ones, x), axis=1)
        return np.matmul(X , theta)
    return None

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape:
        return (y_hat - y) **2
    return None

def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    y = check_dim_vector(y)
    y_hat= check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape:
        tmp = loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
        return np.sum(tmp)
    return None


