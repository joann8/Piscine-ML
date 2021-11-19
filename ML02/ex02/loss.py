import numpy as np

def check_dim_maxtrix(mat):
    if isinstance(mat, np.ndarray):
        if np.ndim(mat) == 1:
            mat = np.reshape(mat, (mat.shape[0], 1))       
        if mat.shape[0] > 0 and mat.shape[1] > 0:
            return mat
    return None

def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same shapes.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same shapes.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        y = check_dim_maxtrix(y)
        y_hat= check_dim_maxtrix (y_hat)
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            tmp = ((y_hat - y) **2) * (1 / (2 * y.shape[0]))
            return np.sum(tmp)
        return None
    except Exception as err:
        print(err)
        return None