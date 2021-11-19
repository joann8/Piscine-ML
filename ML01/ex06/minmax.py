import numpy as np

def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
        x: has to be an numpy.array, a vector.
    Return:
        x’ as a numpy.array.
        None if x is a non-empty numpy.array or not a numpy.array.
        None if x is not of the expected type.
    Raises:
        This function shouldn’t raise any Exception.
    """
    try:
        if isinstance(x, np.ndarray):          
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            print("wrongs inputs")
            return None
    except Exception as err:
        print(err)
        return None