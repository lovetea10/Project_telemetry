import pandas as pd
import os


def read_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Читает данные из CSV файла и возвращает DataFrame.
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


def filter_dataframe(df: pd.DataFrame, filter_type: str = 'any', columns: list[str] = None,
                     inplace: bool = False) -> pd.DataFrame | None:
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


if __name__ == "__main__":
    file_path = 'data.csv'
    try:
        df = read_csv_to_dataframe(file_path)
        print(df)
    except Exception as e:
        print(e)

    filtered_df = filter_dataframe(df, filter_type='any', columns=['column1', 'column2'], inplace=False)

    print(filtered_df)