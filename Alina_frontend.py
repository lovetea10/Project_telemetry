import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QDialog, QLabel, QHBoxLayout, QMessageBox, QLineEdit, QComboBox, QListWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Liza_backend import DataEditor, MathFunctions
import scipy.stats as stats
import re
from sympy import symbols, sympify, lambdify, sin, cos
from PyQt5.QtGui import QDoubleValidator


class MyFigure(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Основной график')
        self.ax.grid(True)

    def plot_data(self, x, y, regression_line=None):
        self.ax.clear()
        self.ax.plot(x, y, 'o', label='Данные')
        if regression_line is not None:
            self.ax.plot(x, regression_line, color='red', label='Регрессия')
        self.ax.set_title('Данные и регрессионная линия')
        self.ax.grid(True)
        self.ax.legend()
        self.draw()

    def plot_polynomial_regression(self, x, y, regression_line):
        self.ax.clear()
        self.ax.plot(x, y, 'o', label='Данные')
        self.ax.plot(x, regression_line, color='red', label='Регрессия по многочлену')
        self.ax.set_title('Регрессия по многочлену')
        self.ax.grid(True)
        self.ax.legend()
        self.draw()

    def plot_scatter(self, x, y, title):
        self.ax.clear()
        self.ax.scatter(x, y, color='blue', marker='o', label='Данные')
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.legend()
        self.draw()


class PolynomialInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ввод многочлена')
        self.layout = QVBoxLayout(self)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Введите многочлен в формате: sin(x)-2*cos(x)+7*x^2+6*x^3+3")
        self.layout.addWidget(self.input_field)

        self.submit_button = QPushButton("Подтвердить", self)
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button)

        self.polynomial = None

    def submit(self):
        text = self.input_field.text()
        if text:
            self.polynomial = text
            QMessageBox.information(self, "Введённый многочлен", f"Вы ввели:\n{text}")
            self.accept()


class InputArrayDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ввод двумерного массива')
        self.setGeometry(600, 400, 500, 400)

        self.layout = QVBoxLayout(self)

        self.x_input_label = QLabel("Введите значения x (через запятую, от 0 до 1):", self)
        self.layout.addWidget(self.x_input_label)
        self.x_input_field = QLineEdit(self)
        self.x_input_field.setPlaceholderText("Пример: 1.0987, 0.00987, 8.0987")
        self.layout.addWidget(self.x_input_field)

        self.y_input_label = QLabel("Введите значения y (через запятую, от 0 до 1):", self)
        self.layout.addWidget(self.y_input_label)
        self.y_input_field = QLineEdit(self)
        self.y_input_field.setPlaceholderText("Пример: 0.5,0.6,0.7,0.8")
        self.layout.addWidget(self.y_input_field)

        self.save_button = QPushButton("Сохранить массив в CSV", self)
        self.save_button.clicked.connect(self.save_and_display)
        self.layout.addWidget(self.save_button)

        self.display_button = QPushButton("Отобразить данные массива", self)
        self.display_button.clicked.connect(self.display_array)
        self.layout.addWidget(self.display_button)

        self.polynomial_button = QPushButton("Ввести многочлены")
        self.polynomial_button.clicked.connect(self.open_polynomial_dialog)
        self.layout.addWidget(self.polynomial_button)

        self.x_values = None
        self.y_values = None
        self.array = None

    def save_and_display(self):
        x_text = self.x_input_field.text()
        y_text = self.y_input_field.text()
        if x_text and y_text:
            try:
                x_values = np.array(list(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', x_text.replace(" ", "")))))
                y_values = np.array(list(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', y_text.replace(" ", "")))))
                if len(x_values) < 2 or len(y_values) < 2:
                    QMessageBox.critical(self, "Ошибка ввода", "Необходимо ввести минимум 2 значения x и 2 значения y.")
                    return
                if len(x_values) != len(y_values):
                    QMessageBox.critical(self, "Ошибка ввода",
                                         "Количество введенных значений x и y должно быть одинаковым.")
                    return
                self.x_values = x_values
                self.y_values = y_values
                array = np.stack((x_values, y_values), axis=1)
                self.array = array
                self.save_array_to_csv(array, 'generated_person.csv')
                self.display_array()
            except ValueError:
                QMessageBox.critical(self, "Ошибка ввода",
                                     "Неверный формат ввода. Пример: 0.1,0.2,0.3,0.4 или 1.0987, 0.00987, 8.0987")

    def display_array(self):
        if self.array is not None:
            self.parent().display_data_on_figure2(self.array,
                                                  "Данные, которые ввел пользователь")  # вызываем метод родительского окна
        else:
            QMessageBox.warning(self, "Предупреждение", "Массив не был введен, или введён с ошибкой.")

    def save_array_to_csv(self, array, file_path):
        header = 'x,y' if array.shape[1] == 2 else None
        try:
            np.savetxt(file_path, array, delimiter=',', fmt='%.5f', header=header, comments='')
            QMessageBox.information(self, "Сохранение массива", f"Массив сохранен в {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Ошибка сохранения в csv: {e}")

    def open_polynomial_dialog(self):
        if self.parent().dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV или введите двумерный массив.")
            return
        dialog = PolynomialInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            print(f"Многочлен введён: {dialog.polynomial}")
            self.parent().polynomial = dialog.polynomial  # Передача многочлена в главное окно


def option_recognition(option, rows, cols):
    if option == "Равномерное":
        return np.random.rand(rows, cols)
    elif option == "Нормальное":
        return np.random.normal(size=(rows, cols))
    elif option == "Пирсон":
        df, loc, scale = 2, 0, 1
        array = stats.chi2.rvs(df, loc=loc, scale=scale, size=(rows, cols))
        return array
    elif option == "Фишер":
        dfn, dfd, loc, scale = 2, 2, 0, 1
        array = stats.f.rvs(dfn, dfd, loc=loc, scale=scale, size=(rows, cols))
        return array
    return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Анализ данных')
        self.setGeometry(100, 100, 1400, 900)

        # Основной виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Графики
        graph_layout = QHBoxLayout()
        self.figure1 = MyFigure()
        self.figure2 = MyFigure()
        graph_layout.addWidget(self.figure1)
        graph_layout.addWidget(self.figure2)
        layout.addLayout(graph_layout)

        # Кнопки
        button_layout = QHBoxLayout()

        self.open_csv_button = QPushButton('Открыть файл CSV')
        self.open_csv_button.clicked.connect(self.open_csv_dialog)  # Обработка загрузки CSV
        button_layout.addWidget(self.open_csv_button)

        self.input_array_button = QPushButton('Ввести двумерный массив')
        self.input_array_button.clicked.connect(self.input_2d_array)  # Обработка ввода массива
        button_layout.addWidget(self.input_array_button)

        self.generate_array_button = QPushButton('Сгенерировать двумерный массив')
        self.generate_array_button.clicked.connect(self.generate_2d_array)  # Обработка генерации массива
        button_layout.addWidget(self.generate_array_button)

        self.confidence_interval_button = QPushButton('Построить доверительный интервал')
        self.confidence_interval_button.clicked.connect(
            self.build_confidence_interval_dialog)  # Обработка доверительного интервала
        button_layout.addWidget(self.confidence_interval_button)

        self.regression_layout = QHBoxLayout()
        self.regression_button1 = QPushButton('Построить регрессию для 1 графика')
        self.regression_button2 = QPushButton('Построить регрессию для 2 графика')
        self.regression_button1.clicked.connect(lambda: self.build_linear_regression(figure_num=1))
        self.regression_button2.clicked.connect(lambda: self.build_linear_regression(figure_num=2))
        self.regression_layout.addWidget(self.regression_button1)
        self.regression_layout.addWidget(self.regression_button2)
        layout.addLayout(self.regression_layout)

        self.extra_button2 = QPushButton('Проверка на мультиколлинеарность')
        self.extra_button2.clicked.connect(
            self.check_multicollinearity_dialog)  # Подключаем функцию проверки на мультиколлинеарность
        button_layout.addWidget(self.extra_button2)

        self.rmse_layout = QHBoxLayout()
        self.rmse_button1 = QPushButton('СКО для 1 графика')
        self.rmse_button2 = QPushButton('СКО для 2 графика')
        self.rmse_button1.clicked.connect(lambda: self.calculate_rmse(figure_num=1))
        self.rmse_button2.clicked.connect(lambda: self.calculate_rmse(figure_num=2))
        self.rmse_layout.addWidget(self.rmse_button1)
        self.rmse_layout.addWidget(self.rmse_button2)
        layout.addLayout(self.rmse_layout)

        self.r_squared_layout = QHBoxLayout()
        self.r_squared_button1 = QPushButton('R² для 1 графика')
        self.r_squared_button2 = QPushButton('R² для 2 графика')
        self.r_squared_button1.clicked.connect(lambda: self.calculate_r_squared(figure_num=1))
        self.r_squared_button2.clicked.connect(lambda: self.calculate_r_squared(figure_num=2))
        self.r_squared_layout.addWidget(self.r_squared_button1)
        self.r_squared_layout.addWidget(self.r_squared_button2)
        layout.addLayout(self.r_squared_layout)

        layout.addLayout(button_layout)
        layout.addStretch()

        central_widget.setLayout(layout)
        self.dataframe = None
        self.polynomial = None  # Поле для хранения многочлена
        self.generated_array = None
        self.data_for_figure1 = None
        self.data_for_figure2 = None

    def input_2d_array(self):
        dialog = InputArrayDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            array = dialog.result
            QMessageBox.information(self, "Введённый массив", f"Вы ввели:\n{array}")

    def generate_2d_array(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор распределения")
        dialog.setGeometry(200, 200, 400, 200)
        dialog_layout = QVBoxLayout(dialog)

        distribution_label = QLabel("Выберите распределение:", dialog)
        dialog_layout.addWidget(distribution_label)

        distribution_combo = QComboBox(dialog)
        distribution_combo.addItems(["Равномерное", "Нормальное", "Пирсон", "Фишер"])
        dialog_layout.addWidget(distribution_combo)

        save_button = QPushButton("Сохранить файл CSV", dialog)
        dialog_layout.addWidget(save_button)

        selected_distribution = ""
        array = None

        def on_save_clicked():
            nonlocal selected_distribution, array
            selected_distribution = distribution_combo.currentText()
            rows, cols = 100, 2
            array = option_recognition(selected_distribution, rows, cols)
            if array is None:
                QMessageBox.critical(self, "Ошибка", "Неизвестный вид распределения.")
            else:
                save_path = 'generated.csv'
                header = 'x,y'
                np.savetxt(save_path, array, delimiter=',', fmt='%.5f', header=header, comments='')
                QMessageBox.information(self, "Сохранение массива",
                                        f"Массив, сгенерированный по распределению: {selected_distribution} сохранен в {save_path}.")
                self.generated_array = array  # Сохраняем сгенерированный массив
            dialog.accept()

        save_button.clicked.connect(on_save_clicked)
        dialog.exec_()

    def open_csv_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Загрузка CSV")
        dialog_layout = QVBoxLayout(dialog)

        load_button1 = QPushButton("Загрузить данные для 1 графика", dialog)
        load_button2 = QPushButton("Загрузить данные для 2 графика", dialog)
        dialog_layout.addWidget(load_button1)
        dialog_layout.addWidget(load_button2)

        def load_data_for_figure1():
            file_path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV files (*.csv)")
            if file_path:
                try:
                    self.data_for_figure1 = DataEditor.read_csv_to_dataframe(file_path).values  # Читаем CSV в массив
                    self.display_data_on_figure1(self.data_for_figure1, "Данные, загруженные из CSV для 1 графика")
                    QMessageBox.information(self, "Информация", f"Данные загружены из {file_path}.")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", str(e))
            dialog.accept()

        def load_data_for_figure2():
            file_path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV files (*.csv)")
            if file_path:
                try:
                    self.dataframe = DataEditor.read_csv_to_dataframe(file_path)
                    self.display_data_on_figure2(self.dataframe.values, "Данные, загруженные из CSV для 2 графика")
                    QMessageBox.information(self, "Информация", f"Данные загружены из {file_path}.")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", str(e))
            dialog.accept()

        load_button1.clicked.connect(load_data_for_figure1)
        load_button2.clicked.connect(load_data_for_figure2)
        dialog.exec_()

    def display_data_on_figure1(self, array, title):
        if array is not None and array.shape[1] == 2:
            x = array[:, 0]
            y = array[:, 1]
            self.figure1.plot_scatter(x, y, title)
        else:
            QMessageBox.warning(self, "Предупреждение", "Невозможно отобразить данные. Неверное количество столбцов.")

    def display_data_on_figure2(self, array, title):
        if array is not None and array.shape[1] == 2:
            x = array[:, 0]
            y = array[:, 1]
            self.figure2.plot_scatter(x, y, title)
        else:
            QMessageBox.warning(self, "Предупреждение", "Невозможно отобразить данные. Неверное количество столбцов.")

    def build_confidence_interval_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор графика")
        dialog_layout = QVBoxLayout(dialog)

        button1 = QPushButton("доверительный интервал для 1 графика", dialog)
        button2 = QPushButton("доверительный интервал для 2 графика", dialog)
        dialog_layout.addWidget(button1)
        dialog_layout.addWidget(button2)

        def build_for_figure1():
            if self.data_for_figure1 is None:
                QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 1 графика.")
                return
            try:
                target_column = "y"
                x = self.data_for_figure1[:, 0]
                y = self.data_for_figure1[:, 1]
                lower_bound, upper_bound = MathFunctions.confidence_interval(pd.DataFrame({'x': x, 'y': y}),
                                                                             target_column)
                QMessageBox.information(self, "Доверительный интервал",
                                        f"Доверительный интервал для '{target_column}': ({lower_bound}, {upper_bound})")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            dialog.accept()

        def build_for_figure2():
            if self.dataframe is None:
                QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 2 графика.")
                return

            target_column = "y"
            try:
                lower_bound, upper_bound = MathFunctions.confidence_interval(self.dataframe, target_column)
                QMessageBox.information(self, "Доверительный интервал",
                                        f"Доверительный интервал для '{target_column}': ({lower_bound}, {upper_bound})")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            dialog.accept()

        button1.clicked.connect(build_for_figure1)
        button2.clicked.connect(build_for_figure2)
        dialog.exec_()

    def check_multicollinearity_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор графика")
        dialog_layout = QVBoxLayout(dialog)

        button1 = QPushButton("мультиколлинеарность для 1 графика", dialog)
        button2 = QPushButton("мультиколлинеарность для 2 графика", dialog)
        dialog_layout.addWidget(button1)
        dialog_layout.addWidget(button2)

        def check_for_figure1():
            if self.data_for_figure1 is None:
                QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 1 графика.")
                return
            try:
                x = self.data_for_figure1[:, 0]
                y = self.data_for_figure1[:, 1]
                target_column = "y"
                multicollinear_features = MathFunctions.check_multicollinearity(pd.DataFrame({'x': x, 'y': y}),
                                                                                target_column)
                if multicollinear_features:
                    QMessageBox.information(self, "Результаты проверки",
                                            f"Обнаруженные мультиколлинеарные параметры: {', '.join(multicollinear_features)}")
                else:
                    QMessageBox.information(self, "Результаты проверки",
                                            "Мультиколлинеарность не обнаружена.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при проверке на мультиколлинеарность: {e}")
            dialog.accept()

        def check_for_figure2():
            if self.dataframe is None:
                QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 2 графика.")
                return
            target_column = "y"
            try:
                multicollinear_features = MathFunctions.check_multicollinearity(self.dataframe, target_column)
                if multicollinear_features:
                    QMessageBox.information(self, "Результаты проверки",
                                            f"Обнаруженные мультиколлинеарные параметры: {', '.join(multicollinear_features)}")
                else:
                    QMessageBox.information(self, "Результаты проверки",
                                            "Мультиколлинеарность не обнаружена.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при проверке на мультиколлинеарность: {e}")
            dialog.accept()

        button1.clicked.connect(check_for_figure1)
        button2.clicked.connect(check_for_figure2)
        dialog.exec_()

    def parse_polynomial(self, polynomial_string):
        """Парсит строку многочлена и возвращает список коэффициентов."""
        polynomial_string = polynomial_string.replace(" ", "")

        # Заменяем ^ на ** для корректной обработки в Python
        polynomial_string = polynomial_string.replace("^", "**")

        # Разделяем многочлен на слагаемые
        terms = re.findall(
            r"([+-]?\s*\d*\.?\d*\s*(?:\*\s*[a-z]+\(\s*[a-z]\s*\))?|\d*\.?\d*\s*(?:\*\s*[a-z]\s*\**\s*\d*)?|[-+]?\d*\.?\d*)",
            polynomial_string)

        coefficients = []
        x = symbols('x')
        for term in terms:
            term = term.replace(" ", "")
            if term:
                try:
                    if 'sin(x)' in term:
                        coeff = float(term.replace('sin(x)', '')) if term.replace('sin(x)', '') else 1.0
                        coefficients.append(coeff)
                    elif 'cos(x)' in term:
                        coeff = float(term.replace('cos(x)', '')) if term.replace('cos(x)', '') else 1.0
                        coefficients.append(coeff)
                    elif 'x**' in term:
                        coeff_str = term.split('*x**')[0]
                        coeff = float(coeff_str) if coeff_str else 1.0
                        coefficients.append(coeff)
                    elif 'x' in term:
                        coeff_str = term.split('*x')[0]
                        coeff = float(coeff_str) if coeff_str else 1.0
                        coefficients.append(coeff)
                    elif re.match(r'[-+]?\d*\.?\d*$', term):  # Это константа
                        coeff = float(term)
                        coefficients.append(coeff)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка при парсинге многочлена {term}: {e}")
                    return None
        return coefficients

    def build_linear_regression(self, figure_num):
        if figure_num == 1 and self.data_for_figure1 is None and self.polynomial is None or figure_num == 2 and self.dataframe is None and self.polynomial is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для этого графика.")
            return
        target_column = "y"
        try:
            if self.polynomial is None:  # Выполнение линейной регрессии, если многочлен не задан
                if figure_num == 1:
                    x = self.data_for_figure1[:, 0]
                    y = self.data_for_figure1[:, 1]
                    coefficients = MathFunctions.calculate_mnk_coefficients(pd.DataFrame({'x': x, 'y': y}), 'y')
                    linear_func = MathFunctions.create_linear_function(coefficients)
                    regression_line = np.array([linear_func(x_val) for x_val in x])
                    self.figure1.plot_data(x, y, regression_line)
                    r_squared = MathFunctions.calculate_r_squared(pd.DataFrame({'x': x, 'y': y}), 'y', regression_line)
                    QMessageBox.information(self, "Результат регрессии",
                                            f"Коэффициенты: {coefficients}, R^2: {r_squared}")
                if figure_num == 2:
                    coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)
                    if len(self.dataframe) <= len(coefficients):
                        raise ValueError("Not enough data for regression.")
                    linear_func = MathFunctions.create_linear_function(coefficients)
                    x_values = self.dataframe.drop(columns=[target_column]).values
                    if x_values.shape[1] > 1:
                        regression_line = np.array([linear_func(*row) for row in x_values])
                    else:
                        regression_line = np.array([linear_func(x) for x in x_values.flatten()])
                    self.figure2.plot_data(x_values.flatten(), self.dataframe[target_column].values, regression_line)
                    r_squared = MathFunctions.calculate_r_squared(self.dataframe, target_column, regression_line)
                    QMessageBox.information(self, "Результат регрессии",
                                            f"Коэффициенты: {coefficients}, R^2: {r_squared}")


            else:  # Выполнение регрессии по многочлену, если задан
                x = symbols('x')
                try:
                    expr = sympify(self.polynomial)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка при построении регрессии: {e}")
                    return

                func = lambdify(x, expr, modules=['numpy',
                                                  {'sin': np.sin, 'cos': np.cos}])
                coefficients = self.parse_polynomial(self.polynomial)
                if coefficients is None:
                    return
                if figure_num == 1:
                    if self.data_for_figure1 is None:
                        QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 1 графика.")
                        return
                    x_values = self.data_for_figure1[:, 0]
                    regression_line = np.array([func(val) for val in x_values])
                    y = self.data_for_figure1[:, 1]
                    self.figure1.plot_polynomial_regression(x_values, y, regression_line)
                    r_squared = MathFunctions.calculate_r_squared(pd.DataFrame({'x': x_values, 'y': y}), 'y',
                                                                  regression_line)
                    QMessageBox.information(self, "Результат регрессии",
                                            f"Регрессия по многочлену, R^2: {r_squared}, Коэффициенты: {coefficients}")

                if figure_num == 2:
                    if self.dataframe is None:
                        QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для 2 графика.")
                        return
                    x_values = self.dataframe.drop(columns=[target_column]).values.flatten()
                    regression_line = np.array([func(val) for val in x_values])
                    self.figure2.plot_polynomial_regression(x_values, self.dataframe[target_column].values,
                                                            regression_line)
                    r_squared = MathFunctions.calculate_r_squared(self.dataframe, target_column, regression_line)
                    QMessageBox.information(self, "Результат регрессии",
                                            f"Регрессия по многочлену, R^2: {r_squared}, Коэффициенты: {coefficients}")

        except (ValueError, KeyError, Exception) as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении регрессии: {e}")

    def calculate_rmse(self, figure_num):
        if figure_num == 1 and self.data_for_figure1 is None or figure_num == 2 and self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для этого графика")
            return
        target_column = "y"
        try:
            if figure_num == 1:
                x = self.data_for_figure1[:, 0]
                y = self.data_for_figure1[:, 1]
                coefficients = MathFunctions.calculate_mnk_coefficients(pd.DataFrame({'x': x, 'y': y}), target_column)
                linear_func = MathFunctions.create_linear_function(coefficients)
                rmse_value = MathFunctions.calculate_rmse(pd.DataFrame({'x': x, 'y': y}), linear_func, target_column)
                QMessageBox.information(self, "Результат", f"Среднеквадратичное отклонение (СКО): {rmse_value:.4f}")
            if figure_num == 2:
                coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)
                linear_func = MathFunctions.create_linear_function(coefficients)
                rmse_value = MathFunctions.calculate_rmse(self.dataframe, linear_func, target_column)
                QMessageBox.information(self, "Результат", f"Среднеквадратичное отклонение (СКО): {rmse_value:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def calculate_r_squared(self, figure_num):
        if figure_num == 1 and self.data_for_figure1 is None or figure_num == 2 and self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для этого графика.")
            return

        target_column = "y"
        try:
            if figure_num == 1:
                x = self.data_for_figure1[:, 0]
                y = self.data_for_figure1[:, 1]
                coefficients = MathFunctions.calculate_mnk_coefficients(pd.DataFrame({'x': x, 'y': y}), target_column)
                linear_func = MathFunctions.create_linear_function(coefficients[::-1])
                x_values = pd.DataFrame({'x': x})
                regression_line = np.array([linear_func(x_val) for x_val in x])
                r_squared_value = MathFunctions.calculate_r_squared(pd.DataFrame({'x': x, 'y': y}), target_column,
                                                                    regression_line)
                QMessageBox.information(self, "Результат", f"Коэффициент детерминации (R²): {r_squared_value:.4f}")

            if figure_num == 2:
                coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)
                linear_func = MathFunctions.create_linear_function(coefficients[::-1])
                x_values = self.dataframe.drop(columns=[target_column]).values
                regression_line = np.array([linear_func(*row) for row in x_values])
                r_squared_value = MathFunctions.calculate_r_squared(self.dataframe, target_column, regression_line)
                QMessageBox.information(self, "Результат", f"Коэффициент детерминации (R²): {r_squared_value:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())