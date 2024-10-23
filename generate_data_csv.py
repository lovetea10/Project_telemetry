import numpy as np
import pandas as pd


def generate_and_save_data():
    np.random.seed(0)
    # Генерируем случайные значения для x
    x = np.random.rand(100)
    # Генерируем y = 2x + шум
    y = 2 * x + np.random.normal(0, 0.1, size=x.shape)

    # Сохраняем в CSV
    data = pd.DataFrame({'x': x, 'y': y})
    file_path = 'generated_data.csv'
    data.to_csv(file_path, index=False)

    return file_path