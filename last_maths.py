from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# Абстрактный класс для регрессионной модели
class RegressionModel(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def mse(self):
        pass
    @abstractmethod
    def print_coefficients(self):
        pass
    @abstractmethod
    def gauss_jordan_inverse(matrix):
        n = len(matrix)
    # Создаем единичную матрицу того же размера
    identity_matrix = np.eye(n)
    # Копируем исходную матрицу, чтобы не изменять её
    augmented_matrix = np.hstack((matrix, identity_matrix))

    for i in range(n):
        # Нормализуем текущую строку
        factor = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / factor

        # Обнуляем все остальные элементы в текущем столбце
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]

    # Возвращаем правую часть расширенной матрицы, которая теперь является обратной
    return augmented_matrix[:, n:]


# Реализация класса RegressionModel для линейной регрессии
class LinearRegressionModel(RegressionModel):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.coefficients = None
        self.y_pred = None

    def add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self):
        X_design = self.add_intercept(self.X)
        self.coefficients = self.gauss_jordan_inverse(X_design.T @ X_design) @ X_design.T @ self.y
        self.y_pred = X_design @ self.coefficients

    def mse(self): #ско
        return np.mean((self.y - self.y_pred) ** 2)

    def print_coefficients(self):
        print("Коэффициенты регрессии:", self.coefficients)


# Реализация класса RegressionModel для полиномиальной регрессии
"""""Реализует методы для добавления константы, полиномиальных признаков, подгонки модели, вычисления СКО,
визуализации результатов и вывода коэффициентов.
"""""
class PolynomialRegressionModel(RegressionModel):
    def __init__(self, X, y, polynomial_terms=None):
        self.X = X
        self.y = y
        self.polynomial_terms = polynomial_terms if polynomial_terms else [(1, 0)]
        self.coefficients = None
        self.y_pred = None

    def generate_polynomial_features(self):
        n_samples = self.X.shape[0]
        X_poly = np.ones((n_samples, 1))

        for coeff, degree in self.polynomial_terms:
            term = np.sum([coeff * (self.X[:, i] ** degree) for i in range(self.X.shape[1])], axis=0)
            X_poly = np.column_stack((X_poly, term))

        return X_poly

    def fit(self):
        X_poly = self.generate_polynomial_features()
        self.coefficients = gauss_jordan_inverse(X_poly.T @ X_poly) @ X_poly.T @ self.y
        self.y_pred = X_poly @ self.coefficients

    def mse(self):
        return np.mean((self.y - self.y_pred) ** 2)

    def print_coefficients(self):
        print("Коэффициенты регрессии:")
        for (coeff, degree), coefficient in zip(self.polynomial_terms, self.coefficients[1:]):
            term_str = f"{coeff} * x^{degree}"
            print(f"{term_str}: {coefficient:.4f}")
        print(f"Свободный член: {self.coefficients[0]:.4f}")


# Абстрактный класс для проверки мультиколлинеарности
class MulticollinearityChecker(ABC):
    @abstractmethod
    def calculate_vif(self):
        pass

    @abstractmethod
    def check(self):
        pass


# Реализация класса MulticollinearityChecker
"""""
Конкретная реализация проверки мультиколлинеарности с использованием VIF (Коэффициент детерминации).
Реализует метод для расчета и вывода коэффициента детерминации.
"""""
class VIFMulticollinearityChecker(MulticollinearityChecker):
    def __init__(self, X):
        self.X = X

    def calculate_vif(self):
        vif = []
        for i in range(self.X.shape[1]):
            X_i = np.delete(self.X, i, axis=1)
            y_i = self.X[:, i]
            X_i_design = np.hstack([np.ones((X_i.shape[0], 1)), X_i])
            coeffs_i =  self.gauss_jordan_inverse(X_i_design.T @ X_i_design) @ X_i_design.T @ y_i
            y_pred_i = X_i_design @ coeffs_i
            r_squared_i = 1 - np.sum((y_i - y_pred_i) ** 2) / np.sum((y_i - np.mean(y_i)) ** 2)
            vif.append(1 / (1 - r_squared_i))
        return vif

    def check(self):
        vif_values = self.calculate_vif()
        print("VIF для каждого признака:", vif_values)

        """""
    # Подзадача 1: Генерация данных и линейная регрессия

    model = PolynomialRegressionModel(X, y)
    model.fit()

    # Подзадача 2: Полиномиальная регрессия
    poly_model.fit()

    # Подзадача 4: Доверительный интервал для ошибок и проверка значимости
    residuals = y_db - model_db.y_pred
    mse_db = np.mean(residuals ** 2)
    n = X_db.shape[0]
    p = X_db.shape[1]
    standard_error = np.sqrt(mse_db / (n - p - 1))

    t_value = stats.t.ppf((1 + confidence) / 2, df=n - p - 1)
    coeff_intervals = [(coeff - t_value * standard_error, coeff + t_value * standard_error) for coeff in model_db.coefficients]

    # Коэффициент детерминации R^2
    ss_total = np.sum((y_db - np.mean(y_db)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"Коэффициент детерминации (R^2): {r_squared:.3f}")

    # Подзадача 5: Проверка мультиколлинеарности
    multicollinearity_checker = VIFMulticollinearityChecker(X_db)
    multicollinearity_checker.check()
    
    #Ввод многочлена в формате: [(2, 3), (9, 3), (6, 6)] - будет многочлен 2x^3+9y^3+6z^6
"""""
        
        