import pandas as pd
import statsmodels.api as sm

def run_regression(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена в DataFrame.")

    Y = df[target_column]
    X = df.drop(columns=[target_column])
    X = sm.add_constant(X)

    if X.shape[0] < X.shape[1]:
        raise ValueError("Недостаточно данных для вычисления коэффициентов.")

    model = sm.OLS(Y, X).fit()
    return model

# Тестирование функции
def test_run_regression():
    # Создание тестового DataFrame
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'target': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    # Успешный тест
    model = run_regression(df, 'target')
    print(model.summary())

    # Тест на отсутствие целевой переменной
    try:
        run_regression(df, 'nonexistent_target')
    except ValueError as e:
        print(e)  # Ожидаемое сообщение об ошибке

    # Тест на недостаток данных
    small_data = {
        'feature1': [1],
        'target': [1]
    }
    small_df = pd.DataFrame(small_data)
    
    try:
        run_regression(small_df, 'target')
    except ValueError as e:
        print(e)  # Ожидаемое сообщение об ошибке

# Запуск тестов
test_run_regression()