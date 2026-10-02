import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate = 0.005, n_epochs = 5000):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.w=None
        self.cost_history = []

    def sigmoid_(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))

    def compute_cost(self, X, y):
        m = X.shape[0]

        y_hat = self.sigmoid_(X @ self.w)

        y_hat = np.clip(y_hat, 1e-15, 1-(1e-15))
        j = (-1/m) * np.sum( y*np.log(y_hat) + (1-y)*np.log(1-y_hat) )
        return j

    def compute_gradient(self,X, y):
        m = X.shape[0]
        y_hat = self.sigmoid_(X@self.w)
        error = (y_hat - y)
        dj_dw = (1/m) * (X.T @ error)
        return dj_dw

    def gradient_descent(self, X, y):
        for i in range(self.epochs):
            self.w = self.w - self.lr * self.compute_gradient(X,y)
            self.cost_history.append(self.compute_cost(X,y))

    def fit(self, X, y):
        m = X.shape[0]
        ones = np.ones((m,1))

        X = np.c_[ones,X]
        n = X.shape[1]

        self.w = np.zeros((n,1))
        self.gradient_descent(X,y)

    def predict(self, X, threshold = 0.5):
        m = X.shape[0]
        ones = np.ones((m,1))

        X = np.c_[ones,X]
        n = X.shape[1]

        y_pred = X @ self.w
        z = self.sigmoid_(y_pred)

        return (z>=threshold).astype(int)

    def accuracy_test(self, y_predict, y_true):
        return np.mean(y_predict == y_true)




