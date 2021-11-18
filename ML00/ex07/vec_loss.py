import numpy as np

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    y = check_dim_vector(y)
    y_hat = check_dim_vector(y_hat)
    if y is not None and y_hat is not None and y.shape == y_hat.shape:
        tmp = ((y_hat - y) **2) * (1 / (2 * y.shape[0]))
        return np.sum(tmp)
    return None