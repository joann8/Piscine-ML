import numpy as np

class MyLinearRegression:
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
            if theta.shape[1] == 1:  
                return theta
        return None

    def check_dim_matrix(self, mat):
        if isinstance(mat, np.ndarray):
            if np.ndim(mat) == 1:
                mat = np.reshape(mat, (mat.shape[0], 1))       
            if mat.shape[0] > 0 and mat.shape[1] > 0:
                return mat
        return None

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        try:
            alpha = self.check_alpha(alpha)
            theta = self.check_theta(theta)
            if theta is not None and alpha is not None\
                and isinstance(max_iter, int) and max_iter > 0:
                self.alpha = alpha
                self.max_iter = max_iter
                self.theta = theta
            else:
                print("Error inputs")
                return None
        except Exception as err:
            print(err)
            return None

    def add_intercept(self, x):
        if isinstance(x,np.ndarray) and len(x) > 0:
            ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
            return np.concatenate((ones, x), axis=1)    
        return None

    def gradient(self, x, y):
        try:
            x = self.check_dim_matrix(x)
            y = self.check_dim_matrix(y)
            if x is not None and y is not None and x.shape[0] == y.shape[0] and y.shape[1] == 1 and self.theta.shape[0] == x.shape[1] + 1:
                X = self.add_intercept(x)
                Xtrans = np.transpose(X)
                X_th_y = np.matmul(X, self.theta) - y
                return ((1 / x.shape[0]) * np.matmul(Xtrans, X_th_y)) # format output? a revoir  
            return None
        except Exception as err:
            print(err)
            return None

    def fit_(self, x, y):
        try:    
            x = self.check_dim_matrix(x)
            y = self.check_dim_matrix(y)
            if x is not None and y is not None and x.shape[0] == y.shape[0] and y.shape[1] == 1 and self.theta.shape[0] == x.shape[1] + 1:
                for i in range(self.max_iter):
                    self.theta = self.theta - self.alpha * self.gradient(x, y)
                return self.theta   
            return None
        except Exception as err:
            print(err)
            return None

    def predict_(self, x):
        try:
            x = self.check_dim_matrix(x)
            if x is not None and self.theta.shape[0] == x.shape[1] + 1:
                return np.matmul(self.add_intercept(x), self.theta) # format OK en colonne?
            return None
        except Exception as err:
            print(err)
            return None
    
    def loss_elem_(self, y, y_hat):
        try:
            y = self.check_dim_matrix(y)
            y_hat= self.check_dim_matrix(y_hat)
            if y is not None and y_hat is not None and y.shape[0] == y_hat.shape[0]:
                return (y_hat - y) ** 2
                #return np.dot(y_hat - y, y_hat - y)
            return None
        except Exception as err:
            print(err)
            return None

    def loss_(self, y, y_hat):
        try:
            y = self.check_dim_matrix(y)
            y_hat= self.check_dim_matrix(y_hat)
            y = y.flatten()
            y_hat = y_hat.flatten()
            if y is not None and y_hat is not None and y.shape == y_hat.shape:
                return float(np.dot(y_hat - y, y_hat - y) * (1 / (2 * y.shape[0])))
            return None
        except Exception as err:
            print(err)
            return None