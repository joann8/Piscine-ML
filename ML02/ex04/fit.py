import numpy as np

def check_dim_matrix(mat):
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

def gradient(x, y, theta):
    try:
        x = check_dim_matrix(x)
        y = check_dim_matrix(y)
        if x is not None and y is not None and x.shape[0] == y.shape[0] and y.shape[1] == 1:
            theta = check_theta(theta, x.shape[1])
            if theta is not None:
                X = add_intercept(x)
                Xtrans = np.transpose(X)
                X_th_y = np.matmul(X, theta) - y
                return ((1 / x.shape[0]) * np.matmul(Xtrans, X_th_y)) # format output? a revoir  
        return None
    except Exception as err:
        print(err)
        return None

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of shape m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of shape m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape (number of features + 1, 1).
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:    
        x = check_dim_matrix(x)
        y = check_dim_matrix(y)
        if x is not None and y is not None and x.shape[0] == y.shape[0] and y.shape[1] == 1:
            theta = check_theta(theta, x.shape[1])
            if theta is not None and isinstance(alpha, float) \
            and isinstance(max_iter, int) and max_iter > 0:
               for i in range(max_iter):
                theta = theta - alpha * gradient(x, y, theta)
            return theta   
        return None
    except Exception as err:
        print(err)
        return None