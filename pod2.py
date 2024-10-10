import numpy as np
import matplotlib.pyplot as plt

def add_polynomial_features(X, degree=2):
    X_poly = X.copy()
    for d in range(2, degree+1):
        X_poly = np.hstack([X_poly, X**d])
    return X_poly

degree = 3
X_poly = add_polynomial_features(X, degree)

X_poly_design = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

#линейная регрессия
coefficients_poly = np.linalg.inv(X_poly_design.T @ X_poly_design) @ X_poly_design.T @ y
y_pred_poly = X_poly_design @ coefficients_poly

#ско
mse_poly = np.mean((y - y_pred_poly) ** 2)

plt.scatter(y, y_pred_poly, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.show()