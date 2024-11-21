import unittest
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Предположим, что ваш код находится в функции, например, optimize_functions
def optimize_functions(df, target_column, functions, loss_function):
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в DataFrame.")

    if df.empty:
        raise ValueError("DataFrame не должен быть пустым.")

    if not isinstance(functions, list) or not all(callable(func) for func in functions):
        raise TypeError("Список функций должен содержать только вызываемые объекты (функции).")

    results = {}

    for func in functions:
        initial_params = np.zeros(len(func.__code__.co_varnames))

        def objective_function(params):
            predictions = df.apply(lambda row: func(*row.values[:-1], *params), axis=1)
            return loss_function(df[target_column].values, predictions)

        result = minimize(objective_function, initial_params)
        results[func.__name__] = result.fun

    return results

class TestOptimizeFunctions(unittest.TestCase):

    def setUp(self):
        # Создаем тестовый DataFrame
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        })

    def test_target_column_not_found(self):
        with self.assertRaises(ValueError) as context:
            optimize_functions(self.df, 'non_existent_column', [lambda x: x], lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2))
        self.assertEqual(str(context.exception), "Целевая колонка 'non_existent_column' не найдена в DataFrame.")

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            optimize_functions(empty_df, 'target', [lambda x: x], lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2))
        self.assertEqual(str(context.exception), "DataFrame не должен быть пустым.")

    def test_functions_not_callable(self):
        with self.assertRaises(TypeError) as context:
            optimize_functions(self.df, 'target', ['not_a_function'], lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2))
        self.assertEqual(str(context.exception), "Список функций должен содержать только вызываемые объекты (функции).")

    def test_functions_optimization(self):
        # Пример функции и простой loss function
        def model_function(x1, x2, a, b):
            return a * x1 + b * x2

        def loss_function(y_true, y_pred):
            return np.sum((y_true - y_pred) ** 2)

        results = optimize_functions(self.df, 'target', [model_function], loss_function)
        self.assertIn('model_function', results)
        self.assertIsInstance(results['model_function'], float)

if __name__ == '__main__':
    unittest.main()
