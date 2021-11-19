import numpy as np

def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
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
            return (x - np.mean(x)) / np.std(x)
        else:
            print("wrongs inputs")
            return None
    except Exception as err:
        print(err)
        return None
