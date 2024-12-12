import unittest
import numpy as np
import pandas as pd
from Liza_backend import MathFunctions
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

class TestMathFunctions(unittest.TestCase):

    def setUp(self):
        self.math_functions = MathFunctions()
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [2, 4, 6, 9],
            'C': [1, 3, 2, 4],
            'target': [1, 2, 3, 4]
        })

    def test_create_linear_function(self):
        params = np.array([4.7, 0.6, 1.0, -4.1])
        linear_function = self.math_functions.create_linear_function(params)
        self.assertEqual(linear_function(1, 2, 3), 4.7 + 0.6 * 1 + 1.0 * 2 - 4.1 * 3)

    def test_calculate_mnk_coefficients(self):
        coefficients = self.math_functions.calculate_mnk_coefficients(self.df, 'target')
        self.assertEqual(len(coefficients), 4)

    def test_calculate_rmse(self):
        params = np.array([4.7, 0.6, 1.0, -4.1])
        linear_function = self.math_functions.create_linear_function(params)
        rmse = self.math_functions.calculate_rmse(self.df, linear_function, 'target')
        self.assertGreaterEqual(rmse, 0)

    def test_regression_with_loss_function(self):
        def loss_function(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        functions = [lambda x, y, z, a: x + y + z + a,
                     lambda x, y, z, a: x * y * z * a]

        result = self.math_functions.regression_with_loss_function(self.df, 'target', functions, loss_function)
        self.assertEqual(len(result), 2)

    def test_confidence_interval(self):
        interval = self.math_functions.confidence_interval(self.df, 'target')
        self.assertEqual(len(interval), 2)

    def test_calculate_r_squared(self):
        params = np.array([4.7, 0.6, 1.0, -4.1])
        linear_function = self.math_functions.create_linear_function(params)
        y_pred = np.array([linear_function(1, 2, 3), linear_function(2, 4, 6), linear_function(3, 6, 9), linear_function(4, 9, 4)])
        r_squared = self.math_functions.calculate_r_squared(self.df, 'target', y_pred)
        self.assertGreaterEqual(r_squared, 0)

    def test_f_test_regression_significance(self):
        params = np.array([4.7, 0.6, 1.0, -4.1])
        linear_function = self.math_functions.create_linear_function(params)
        y_pred = np.array([linear_function(1, 2, 3), linear_function(2, 4, 6), linear_function(3, 6, 9), linear_function(4, 9, 4)])
        result = self.math_functions.f_test_regression_significance(self.df, 'target', y_pred, 0.05)
        self.assertIn('F-statistic', result)
        self.assertIn('p-value', result)
        self.assertIn('Significant', result)

if __name__ == '__main__':
    unittest.main()
