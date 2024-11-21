import pandas as pd
import numpy as np
import os
import sqlite3
from sqlalchemy import create_engine
from pymongo import MongoClient
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as stats
from typing import Union, Callable

class DataEditor:
    def read_csv_to_dataframe(file_path: str) -> pd.DataFrame:
        """
        Читает данные из CSV файла и возвращает DataFrame.

        :param file_path: Путь к CSV файлу (строка). Может быть как абсолютным, так и относительным.
        :return: DataFrame с данными из файла
        """

        absolute_path = os.path.abspath(file_path)

        try:
            df = pd.read_csv(absolute_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: '{absolute_path}'. Убедитесь, что путь указан правильно.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Файл '{absolute_path}' пуст. Пожалуйста, проверьте содержимое файла.")
        except pd.errors.ParserError:
            raise ValueError(f"Ошибка парсинга файла '{absolute_path}'. Возможно, файл имеет неверный формат.")
        except PermissionError:
            raise PermissionError(f"Нет доступа к файлу '{absolute_path}'. Проверьте права доступа.")
        except Exception as e:
            raise Exception(f"Произошла ошибка при чтении файла '{absolute_path}': {str(e)}")

    def get_data_from_db(db_type: str, **kwargs) -> pd.DataFrame:
        """
        Получение данных из базы данных.

        Параметры:
            db_type (str): Тип базы данных ('sqlite', 'postgres', 'mysql', 'mongodb').
            **kwargs: Дополнительные параметры в зависимости от типа базы данных.

        Для каждого типа базы данных:

        1. SQLite:
            - db_path (str): Путь к файлу базы данных.
            - query (str): SQL-запрос для выполнения.

        Пример:
            df = get_data_from_db('sqlite', db_path='my_database.db', query='SELECT * FROM my_table')

        2. PostgreSQL:
            - user (str): Имя пользователя.
            - password (str): Пароль.
            - host (str): Хост (например, 'localhost').
            - port (str): Порт (по умолчанию '5432').
            - dbname (str): Имя базы данных.
            - query (str): SQL-запрос для выполнения.

        Пример:
            df = get_data_from_db('postgres', user='user', password='pass', host='localhost', port='5432', dbname='my_db', query='SELECT * FROM my_table')

        3. MySQL:
            - user (str): Имя пользователя.
            - password (str): Пароль.
            - host (str): Хост (например, 'localhost').
            - port (str): Порт (по умолчанию '3306').
            - dbname (str): Имя базы данных.
            - query (str): SQL-запрос для выполнения.

        Пример:
            df = get_data_from_db('mysql', user='user', password='pass', host='localhost', port='3306', dbname='my_db', query='SELECT * FROM my_table')

        4. MongoDB:
            - connection_string (str): Строка подключения к MongoDB.
            - database (str): Имя базы данных.
            - collection (str): Имя коллекции для выборки данных.

        Пример:
            df = get_data_from_db('mongodb', connection_string='mongodb://localhost:27017/', database='my_db', collection='my_collection')

        Возвращает:
            pd.DataFrame: Данные, полученные из базы данных в виде DataFrame.
        """

        try:
            if db_type == 'sqlite':
                conn = sqlite3.connect(kwargs['db_path'])
                df = pd.read_sql_query(kwargs['query'], conn)
                conn.close()

            elif db_type == 'postgres':
                engine = create_engine(
                    f"postgresql://{kwargs['user']}:{kwargs['password']}@{kwargs['host']}:{kwargs['port']}/{kwargs['dbname']}")
                df = pd.read_sql_query(kwargs['query'], engine)

            elif db_type == 'mysql':
                engine = create_engine(
                    f"mysql+pymysql://{kwargs['user']}:{kwargs['password']}@{kwargs['host']}:{kwargs['port']}/{kwargs['dbname']}")
                df = pd.read_sql_query(kwargs['query'], engine)

            elif db_type == 'mongodb':
                client = MongoClient(kwargs['connection_string'])
                data = list(client[kwargs['database']][kwargs['collection']].find())
                df = pd.DataFrame(data)

            else:
                raise ValueError("Unsupported database type")

            return df

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        except (ValueError, KeyError) as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return None

    def save_dataframe_to_db(df: pd.DataFrame, db_type: str, **kwargs) -> None:
        """
    Сохранение данных в базу данных.

    Параметры:
        df (pd.DataFrame): DataFrame, который нужно сохранить.
        db_type (str): Тип базы данных ('sqlite', 'postgres', 'mysql', 'mongodb').
        **kwargs: Дополнительные параметры в зависимости от типа базы данных.

    Для каждого типа базы данных:

    1. SQLite:
        - db_path (str): Путь к файлу базы данных.

    Пример:
        save_dataframe_to_db(df, 'sqlite', db_path='my_database.db')

    2. PostgreSQL:
        - user (str): Имя пользователя.
        - password (str): Пароль.
        - host (str): Хост (например, 'localhost').
        - port (str): Порт (по умолчанию '5432').
        - dbname (str): Имя базы данных.

    Пример:
        save_dataframe_to_db(df, 'postgres', user='user', password='pass', host='localhost', port='5432', dbname='my_db')

    3. MySQL:
        - user (str): Имя пользователя.
        - password (str): Пароль.
        - host (str): Хост (например, 'localhost').
        - port (str): Порт (по умолчанию '3306').
        - dbname (str): Имя базы данных.

    Пример:
        save_dataframe_to_db(df, 'mysql', user='user', password='pass', host='localhost', port='3306', dbname='my_db')

    4. MongoDB:
        - connection_string (str): Строка подключения к MongoDB.
        - database (str): Имя базы данных.
        - collection (str): Имя коллекции для сохранения данных.

    Пример:
        save_dataframe_to_db(df, 'mongodb', connection_string='mongodb://localhost:27017/', database='my_db', collection='my_collection')

    Возвращает:
        None: Данные сохраняются в указанной базе данных.
    """
        try:
            if db_type == 'sqlite':
                db_path = kwargs['db_path']
                engine = create_engine(f'sqlite:///{db_path}')
                df.to_sql('table_name', con=engine, index=False, if_exists='replace')

            elif db_type == 'postgres':
                user = kwargs['user']
                password = kwargs['password']
                host = kwargs['host']
                port = kwargs['port']
                dbname = kwargs['dbname']
                engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
                df.to_sql('table_name', con=engine, index=False, if_exists='replace')

            elif db_type == 'mysql':
                user = kwargs['user']
                password = kwargs['password']
                host = kwargs['host']
                port = kwargs['port']
                dbname = kwargs['dbname']
                engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}')
                df.to_sql('table_name', con=engine, index=False, if_exists='replace')

            elif db_type == 'mongodb':
                connection_string = kwargs['connection_string']
                database_name = kwargs['database']
                collection_name = kwargs['collection']

                client = MongoClient(connection_string)
                db = client[database_name]
                collection = db[collection_name]

                data_dict = df.to_dict("records")
                collection.delete_many({})
                collection.insert_many(data_dict)

            else:
                raise ValueError("Unsupported db_type. Choose from 'sqlite', 'postgres', 'mysql', 'mongodb'.")

        except KeyError as e:
            print(f"Missing required parameter: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def filter_dataframe(df: pd.DataFrame, filter_type: str = 'any', columns: list = None, inplace: bool = False) -> \
    Union[pd.DataFrame, None]:
        """
        Фильтрует DataFrame по заданным критериям.

        :param df: DataFrame для фильтрации
        :param filter_type: Тип фильтрации ('any' или 'all') (строка)
        :param columns: Список названий колонок для фильтрации (список строк или None)
        :param inplace: Если True, изменения применяются к исходному DataFrame (логическое значение)
        :return: Отфильтрованный DataFrame (если inplace=False) или None (если inplace=True)
        """
        if columns is not None:
            if filter_type == 'any':
                mask = df[columns].notna().any(axis=1)
            elif filter_type == 'all':
                mask = df[columns].notna().all(axis=1)
            else:
                raise ValueError("filter_type должен быть 'any' или 'all'")

            filtered_df = df[mask]
        else:
            filtered_df = df.dropna(how='any' if filter_type == 'any' else 'all')

        if inplace:
            df[:] = filtered_df.values
            return None
        else:
            return filtered_df


class MathFunctions:
    def create_linear_function(params: np.ndarray) -> Callable:
        """
        Создает функцию линейной регрессии на основе заданных параметров модели.

        Параметры:
        params (array-like): Коэффициенты модели линейной регрессии, где последний элемент
                             - свободный член, а остальные элементы - коэффициенты
                             для независимых переменных.

        Возвращает:
        Callable: Функция, которая принимает значения независимых переменных и возвращает
                  предсказанное значение зависимой переменной.
        """

        if len(params) == 0:
            raise ValueError("Параметры модели не могут быть пустыми.")

        def linear_function(*args):
            if len(args) != len(params) - 1:
                raise ValueError(f"Количество аргументов должно быть {len(params) - 1}, но получено {len(args)}.")

            return params[-1] + np.sum(k * x for k, x in zip(params[:-1], args))

        return linear_function

    def calculate_mnk_coefficients(df: pd.DataFrame, target_column: str) -> np.ndarray:
        """
        Вычисляет коэффициенты метода наименьших квадратов (МНК) для заданного DataFrame.

        Аргументы:
        df : pd.DataFrame
            DataFrame, содержащий данные, где одна из колонок является целевой переменной,
            а остальные — предикторами.

        target_column : str
            Название колонки, содержащей целевую переменную.

        Возвращает:
        coefficients : pd.Series
            Коэффициенты модели МНК, включая свободный член (константу).
        """

        if target_column not in df.columns:
            raise ValueError(f"Целевая переменная '{target_column}' не найдена в DataFrame.")

        Y = df[target_column]

        X = df.drop(columns=[target_column])

        X = sm.add_constant(X)

        if X.shape[0] < X.shape[1]:
            raise ValueError("Недостаточно данных для вычисления коэффициентов.")

        model = sm.OLS(Y, X).fit()

        return model.params

    def calculate_rmse(df: pd.DataFrame, model_function: Callable, target_column: str):
        """
        Вычисляет среднеквадратичное отклонение (СКО) для линейной регрессии.

        Параметры:
        df : pd.DataFrame
            DataFrame, где одна из колонок - это y, а остальные - это x.
        model_function : Callable
            Функция, которая принимает значения x и возвращает предсказанное значение y.
        target_column : str
            Название колонки с целевыми значениями y.

        Возвращает:
        rmse : float
            Значение среднеквадратичного отклонения.
        """
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

    def regression_with_loss_function(df: pd.DataFrame, target_column: str, functions: list,
                                      loss_function: Callable) -> dict:
        """
        Строит регрессию на основе заданных функций и минимизирует функцию потерь.

        Параметры:
        df : pd.DataFrame
            DataFrame, содержащий данные для регрессии.
        target_column : str
            Название колонки с целевыми значениями y.
        functions : list[Callable]
            Список функций, которые будут применены к входным данным.
        loss_function : Callable
            Функция потерь для минимизации.

        Возвращает:
        loss_functions : dict
            Словарь с функциями и минимизированными значениями функции потерь.
        """

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

    def confidence_interval(df: pd.DataFrame, target_column: str, confidence_level: float = 0.95) -> tuple:
        """
        Вычисляет доверительный интервал для стандартного отклонения зависимой переменной.

        :param df: pandas DataFrame, содержащий данные с зависимой переменной.
        :param target_column: строка, имя столбца зависимой переменной.
        :param confidence_level: уровень доверия (по умолчанию 0.95). Должен быть в диапазоне (0, 1).
        :return: кортеж (нижняя граница, верхняя граница) доверительного интервала для стандартного отклонения.

        :raises ValueError: если target_column не существует в DataFrame или если confidence_level не в диапазоне (0, 1).
        """

        if target_column not in df.columns:
            raise ValueError(f"Столбец '{target_column}' не найден в DataFrame.")

        if not (0 < confidence_level < 1):
            raise ValueError("Уровень доверия должен быть в диапазоне (0, 1).")

        data = df[target_column].dropna()

        n = len(data)
        if n < 2:
            raise ValueError(
                "Недостаточно данных для расчета доверительного интервала (необходимы как минимум 2 наблюдения).")

        std_dev = np.std(data, ddof=1)

        std_error = std_dev / np.sqrt(n)

        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha / 2, df=n - 1)

        margin_of_error = critical_value * std_error
        lower_bound = std_dev - margin_of_error
        upper_bound = std_dev + margin_of_error

        return lower_bound, upper_bound

    def calculate_r_squared(df: pd.DataFrame, target_column: str, regression: list) -> float:
        """
        Рассчитывает коэффициент детерминации (R²) по формуле.

        :param df: DataFrame с данными.
        :param target_column: Название колонки с целевой переменной.
        :param regression: Список значений независимых переменных.
        :return: Коэффициент детерминации (R²).
        """

        y = df[target_column].values

        y_pred = np.array(regression)

        ss_total = np.sum((y - np.mean(y)) ** 2)  # Общая сумма квадратов
        ss_residual = np.sum((y - y_pred) ** 2)  # Остаточная сумма квадратов

        r_squared = 1 - (ss_residual / ss_total)

        return r_squared


import numpy as np
import pandas as pd
from scipy import stats


def f_test_regression_significance(df: pd.DataFrame, target_column: str, regression: list, alpha: float) -> dict:
    """
    Проверяет значимость регрессии по критерию Фишера.

    :param df: DataFrame с данными.
    :param target_column: Название колонки с целевой переменной.
    :param regression: Список значений независимых переменных.
    :param alpha: Уровень значимости.
    :return: Словарь с результатами теста.
    """

    if target_column not in df.columns:
        return {'Error': f'Целевая переменная "{target_column}" не найдена в DataFrame.'}

    if df.empty:
        return {'Error': 'DataFrame пуст.'}

    y = df[target_column].values

    if len(y) == 0:
        return {'Error': f'Целевая переменная "{target_column}" пуста.'}

        y_pred = np.array(regression)

    if len(X) != len(y):
        return {'Error': 'Количество наблюдений в независимых переменных и целевой переменной не совпадает.'}

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_regression = np.sum((y_pred - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    n = len(y)
    k = X.shape[1] - 1

    if n <= k:
        return {'Error': 'Недостаточное количество наблюдений для расчета F-статистики.'}

    ms_regression = ss_regression / k
    ms_residual = ss_residual / (n - k - 1)

    f_statistic = ms_regression / ms_residual

    p_value = 1 - stats.f.cdf(f_statistic, dfn=k, dfd=n - k - 1)

    significant = p_value < alpha

    return {
        'F-statistic': f_statistic,
        'p-value': p_value,
        'Significant': significant
    }


if __name__ == "__main__":
    file_path = 'data.csv'
    try:
        df = DataEditor.read_csv_to_dataframe(file_path)
        print(df)
    except Exception as e:
        print(e)
    print(df)
    y = MathFunctions.create_linear_function([4.7, 0.6, -4.1])
    print(MathFunctions.calculate_rmse(df, y, "Y"))