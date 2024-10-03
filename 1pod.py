import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_samples = 100
X = np.random.uniform(-1, 1, (n_samples, 5))
y = 2 + 3*X[:, 0] - 2*X[:, 1] + X[:, 2] - 2*X[:, 3] + X[:, 4] + np.random.normal(0, 3, n_samples)

X_design = np.hstack([np.ones((X.shape[0], 1)), X])

#мнк
coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
y_pred = X_design @ coefficients

#ско
mse = np.mean((y - y_pred) ** 2)


plt.scatter(y, y_pred, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.show()
