import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine
from pymongo import MongoClient
import statsmodels.api as sm
from typing import Union


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


def filter_dataframe(df: pd.DataFrame, filter_type: str = 'any', columns: list[str] = None, inplace: bool = False) -> Union[pd.DataFrame, None]:
    """
    Фильтрует DataFrame по заданным критериям.
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
        df[:] = filtered_df.values  # Изменяем исходный DataFrame
        return None
    else:
        return filtered_df

def save_dataframe_to_db(df: pd.DataFrame, db_type: str, **kwargs) -> None:
    """
Сохранение данных в базу данных.
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

            # Convert DataFrame to a list of dictionaries
            data_dict = df.to_dict("records")
            collection.delete_many({})  # Clear existing data
            collection.insert_many(data_dict)

        else:
            raise ValueError("Unsupported db_type. Choose from 'sqlite', 'postgres', 'mysql', 'mongodb'.")

    except KeyError as e:
        print(f"Missing required parameter: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_mnk_coefficients(df: pd.DataFrame, target_column: str):
    """
    Вычисляет коэффициенты метода наименьших квадратов (МНК) для заданного DataFrame.

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

if __name__ == "__main__":
    file_path = 'data.csv'
    try:
        df = read_csv_to_dataframe(file_path)
        print(df)
    except Exception as e:
        print(e)
    print(df)