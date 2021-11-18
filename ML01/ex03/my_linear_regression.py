import numpy as np





class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    
    def check_alpha(self, alpha):
        if isinstance(alpha, float) and alpha >= 0 and alpha <= 1:
            return alpha
        return None  
    
    def check_theta(self, theta):
        if isinstance(theta, list):
            theta = np.array(theta)
        if isinstance(theta,np.ndarray) and theta.shape[0] > 0:
            if np.ndim(theta) == 1:
                theta = np.reshape(theta, (theta.shape[0], 1))  
            if theta.shape[0] == 2 and theta.shape[1] == 1:
                return theta 
        return None
       
    def check_dim_vector(self, vect):
        if isinstance(vect, np.ndarray):
            if np.ndim(vect) == 1:
                vect = np.reshape(vect, (vect.shape[0], 1))       
            return vect
        return None

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        alpha = self.check_alpha(alpha)
        theta = self.check_theta(theta)    
        if theta is not None and alpha is not None\
            and isinstance(max_iter, int) and max_iter > 0:
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
        else:
            print("Error inputs")
            quit()
    
    def gradient(self, x, y):
        x = self.check_dim_vector(x)
        y = self.check_dim_vector(y)
        if x is not None and y is not None \
            and x.shape == y.shape and y.shape[1] == 1:
            ones = np.ones((x.shape[0], 1)) 
            X = np.concatenate((ones, x), axis=1)
            X_th_y = np.matmul(X , self.theta) - y
            Xtrans= np.matrix.transpose(X)
            return ((1 / x.shape[0]) * np.matmul(Xtrans, X_th_y))
        return None

    def fit_(self, x, y):
        x = self.check_dim_vector(x)
        y = self.check_dim_vector(y)   
        if x is not None and y is not None \
            and x.shape == y.shape and y.shape[1] == 1:
            for i in range(self.max_iter):
                self.theta = self.theta - self.alpha * self.gradient(x, y)
            return self.theta   
        return None

    def predict_(self, x):
        x = self.check_dim_vector(x)
        if x is not None :
            ones = np.ones((x.shape[0], 1)) 
            X = np.concatenate((ones, x), axis=1)
            return np.matmul(X , self.theta)
        return None
    
    def loss_elem_(self, y, y_hat):
        y = self.check_dim_vector(y)
        y_hat= self.check_dim_vector(y_hat)
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            return (y_hat - y) **2
        return None

    def loss_(self, y, y_hat):
        y = self.check_dim_vector(y)
        y_hat= self.check_dim_vector(y_hat)
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            tmp = self.loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
            return np.sum(tmp)
        return None





    