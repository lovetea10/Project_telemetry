import numpy as np

class Regression:

    def __init__(self, X, y, degree=1):
        self.X = X
        self.y = y
        self.degree = degree
        self.coefficients = None
        self.y_pred = None

    def add_const(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def add_poly(self):
        X_poly = self.X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack([X_poly, self.X ** d])
        return X_poly

    def construction(self):
        if self.degree > 1:
            X_design = self.add_poly()
        else:
            X_design = self.X
        X_design = self.add_const(X_design)
        self.coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ self.y
        self.y_pred = X_design @ self.coefficients

    def mse(self):
        return np.mean((self.y - self.y_pred) ** 2)