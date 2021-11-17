import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x,np.ndarray) and len(x) > 0:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        return np.concatenate((ones, x), axis=1)    
    return None
