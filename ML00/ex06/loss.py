import numpy as np

def check_dim_vector(vect):
    if np.ndim(vect) == 1:
        vect = np.reshape(vect, (vect.shape[0], 1))
        return vect
    if np.ndim(vect) == 2:
        return vect
    return None
    
def predict_(x, theta):
    if isinstance(x,np.ndarray) and len(x) > 0 and \
        isinstance(theta,np.ndarray) and len(theta) == 2:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.reshape(x, (x.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
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
    if isinstance(y,np.ndarray) and isinstance(y_hat, np.ndarray):
        print("y = " , y.shape)
        print("y_hat = " , y_hat.shape)
        y = check_dim_vector(y)
        if y is not None:
            print("new y = " , y.shape)
        else:
            return None
    #     if y.shape[0] == 1 or y.shape[1] == 1:
    #         if y.shape[0] > 1:
    #             Y = np.reshape(y, (y.shape[0], 1))
    #         else:
    #             Y = y
    #     else:
    #         print("Y no a vector")
        
    #     if y_hat.shape[0] == 1 or y_hat.shape[1] == 1:
    #         if y_hat.shape[0] > 1:
    #             Y_hat= np.reshape(y_hat, (y_hat.shape[0], 1))
    #         else:
    #             Y_hat = y_hat
    #     else:
    #         print("Y_hat no a vector")
    # else:
    #     print("Wrongs Inputs Type")
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
    if isinstance(y,np.ndarray) and isinstance(y_hat, np.ndarray):
        pass

