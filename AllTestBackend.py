import unittest
import pandas as pd
import os
import tempfile
from typing import Union
from EngPrak import DataEditor 

class TestReadCSVToDataFrame(unittest.TestCase):

    def setUp(self):
        self.valid_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.valid_csv.write(b'col1,col2\n1,2\n3,4\n')
        self.valid_csv.close()

        self.empty_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.empty_csv.close()

        self.invalid_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.invalid_csv.write(b'col1;col2\n1;2\n3;4\n')
        self.invalid_csv.close()

    def tearDown(self):
        os.remove(self.valid_csv.name)
        os.remove(self.empty_csv.name)
        os.remove(self.invalid_csv.name)

    def test_read_valid_csv(self):
        df = DataEditor.read_csv_to_dataframe(self.valid_csv.name)
        self.assertEqual(df.shape, (2, 2))  # Ожидаем 2 строки и 2 столбца
        self.assertEqual(list(df.columns), ['col1', 'col2'])

    def test_read_empty_csv(self):
        with self.assertRaises(ValueError) as context:
            DataEditor.read_csv_to_dataframe(self.empty_csv.name)
        self.assertEqual(str(context.exception), f"Файл '{os.path.abspath(self.empty_csv.name)}' пуст. Пожалуйста, проверьте содержимое файла.")

    def test_read_invalid_csv(self):
        with self.assertRaises(ValueError) as context:
            DataEditor.read_csv_to_dataframe(self.invalid_csv.name)
        self.assertTrue("Ошибка парсинга файла" in str(context.exception))

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as context:
            DataEditor.read_csv_to_dataframe('non_existent_file.csv')
        self.assertTrue("Файл не найден" in str(context.exception))

    def test_permission_error(self):
        # права только для чтения
        permission_denied_file = tempfile.NamedTemporaryFile(delete=False)
        os.chmod(permission_denied_file.name, 0o000)  
        try:
            with self.assertRaises(PermissionError) as context:
                DataEditor.read_csv_to_dataframe(permission_denied_file.name)
            self.assertTrue("Нет доступа к файлу" in str(context.exception))
        finally:
            os.remove(permission_denied_file.name)  

class TestFilterDataFrame(unittest.TestCase):
    def setUp(self):
        # Создаем пример DataFrame для тестов
        self.df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, None, None, 4]
        })

    def test_filter_any(self):
        result = DataEditor.filter_dataframe(self.df, filter_type='any', columns=['A', 'B'])
        expected = pd.DataFrame({
            'A': [1, 2, 4],
            'B': [None, 2, 4],
            'C': [1, None, 4]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_filter_all(self):
        result = DataEditor.filter_dataframe(self.df, filter_type='all', columns=['A', 'B'])
        expected = pd.DataFrame({
            'A': [4],
            'B': [4],
            'C': [4]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_filter_no_columns(self):
        result = DataEditor.filter_dataframe(self.df, filter_type='any')
        expected = pd.DataFrame({
            'A': [1, 2, 4],
            'B': [None, 2, 4],
            'C': [1, None, 4]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_inplace_filter(self):
        DataEditor.filter_dataframe(self.df, filter_type='any', columns=['A', 'B'], inplace=True)
        expected = pd.DataFrame({
            'A': [1, 2, 4],
            'B': [None, 2, 4],
            'C': [1, None, 4]
        })
        pd.testing.assert_frame_equal(self.df.reset_index(drop=True), expected.reset_index(drop=True))

    def test_invalid_filter_type(self):
        with self.assertRaises(ValueError):
            DataEditor.filter_dataframe(self.df, filter_type='invalid', columns=['A', 'B'])

if __name__ == '__main__':
    unittest.main()
