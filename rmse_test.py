import unittest
import pandas as pd
import numpy as np

def calculate_rmse(df, target_column, model_function):
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в DataFrame.")

    if df.empty:
        raise ValueError("DataFrame не должен быть пустым.")

    if not callable(model_function):
        raise TypeError("model_function должна быть вызываемым объектом (функцией).")

    y_actual = df[target_column].values
    x_values = df.drop(columns=[target_column]).values

    y_predicted = np.array([model_function(*x) for x in x_values])

    rmse = np.sqrt(np.mean(np.power(y_actual - y_predicted, 2)))
    return rmse

class TestCalculateRMSE(unittest.TestCase):

    def setUp(self):
        # Создаем тестовый DataFrame
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        })

    def test_target_column_not_found(self):
        with self.assertRaises(ValueError) as context:
            calculate_rmse(self.df, 'non_existent_column', lambda x, y: x + y)
        self.assertEqual(str(context.exception), "Целевая колонка 'non_existent_column' не найдена в DataFrame.")

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            calculate_rmse(empty_df, 'target', lambda x, y: x + y)
        self.assertEqual(str(context.exception), "DataFrame не должен быть пустым.")

    def test_model_function_not_callable(self):
        with self.assertRaises(TypeError) as context:
            calculate_rmse(self.df, 'target', 'not_a_function')
        self.assertEqual(str(context.exception), "model_function должна быть вызываемым объектом (функцией).")

    def test_rmse_calculation(self):
        # Простой модельный функция, которая возвращает сумму двух признаков
        def model_function(x1, x2):
            return x1 + x2

        expected_rmse = np.sqrt(np.mean(np.power(self.df['target'].values - np.array([model_function(row[0], row[1]) for row in self.df[['feature1', 'feature2']].values]), 2)))
        actual_rmse = calculate_rmse(self.df, 'target', model_function)
        self.assertAlmostEqual(actual_rmse, expected_rmse)

if __name__ == '__main__':
    unittest.main()
