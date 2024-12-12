import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QDialog, QLabel, QHBoxLayout, QMessageBox, QLineEdit, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Liza_backend import DataEditor, MathFunctions
import scipy.stats as stats


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


class InputArrayDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ввод двумерного массива')
        self.layout = QVBoxLayout(self)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Введите массив в формате: [a,b,c;d,e,f]")
        self.layout.addWidget(self.input_field)

        self.submit_button = QPushButton("Отправить", self)
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button)

        self.result = None

    def submit(self):
        text = self.input_field.text()
        if text:
            try:
                # Убираем квадратные скобки и разбиваем на строки
                if text.startswith('[') and text.endswith(']'):
                    text = text[1:-1]
                array = np.array([list(map(float, row.split(','))) for row in text.split('],[')])
                self.result = array
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка ввода", "Неверный формат ввода.")


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

        # Выбор распределений
        self.distribution1_combo = QComboBox()
        self.distribution2_combo = QComboBox()
        self.distribution1_combo.addItems(["Нормальное", "Равномерное"])
        self.distribution2_combo.addItems(["Нормальное", "Равномерное"])

        distribution_layout = QHBoxLayout()
        distribution_layout.addWidget(QLabel("Выберите распределение 1:"))
        distribution_layout.addWidget(self.distribution1_combo)
        distribution_layout.addWidget(QLabel("Выберите распределение 2:"))
        distribution_layout.addWidget(self.distribution2_combo)
        layout.addLayout(distribution_layout)

        # Кнопки
        self.update_button1 = QPushButton('Обновить график 1')
        self.update_button2 = QPushButton('Обновить график 2')
        self.update_button1.clicked.connect(self.update_graph1)
        self.update_button2.clicked.connect(self.update_graph2)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.update_button1)
        button_layout.addWidget(self.update_button2)
        layout.addLayout(button_layout)

        self.open_csv_button = QPushButton('Открыть файл CSV')
        self.open_csv_button.clicked.connect(self.load_csv_data)  # Обработка загрузки CSV
        button_layout.addWidget(self.open_csv_button)

        self.input_array_button = QPushButton('Ввести двумерный массив')
        self.input_array_button.clicked.connect(self.input_2d_array)  # Обработка ввода массива
        button_layout.addWidget(self.input_array_button)

        self.generate_array_button = QPushButton('Сгенерировать двумерный массив')
        self.generate_array_button.clicked.connect(self.generate_2d_array)  # Обработка генерации массива
        button_layout.addWidget(self.generate_array_button)

        self.confidence_interval_button = QPushButton('Построить доверительный интервал')
        self.confidence_interval_button.clicked.connect(
            self.build_confidence_interval)  # Обработка доверительного интервала
        button_layout.addWidget(self.confidence_interval_button)

        self.extra_button1 = QPushButton('Построить линейную регрессию')
        self.extra_button2 = QPushButton('Проверка на мультиколлинеарность')
        self.extra_button2 = QPushButton('Проверка на мультиколлинеарность')
        self.extra_button2.clicked.connect(self.check_multicollinearity)  # Подключаем функцию проверки на мультиколлинеарность

        self.extra_button1.clicked.connect(self.build_linear_regression)  # Подключаем функцию линейной регрессии
        button_layout.addWidget(self.extra_button1)
        button_layout.addWidget(self.extra_button2)

        self.rmse_button = QPushButton('Вычислить СКО')
        self.r_squared_button = QPushButton('Рассчитать R²')

        self.rmse_button.clicked.connect(self.calculate_rmse)  # Добавляем обработчик для СКО
        self.r_squared_button.clicked.connect(self.calculate_r_squared)  # Добавляем обработчик для R²

        button_layout.addWidget(self.rmse_button)
        button_layout.addWidget(self.r_squared_button)

        layout.addStretch()

        central_widget.setLayout(layout)
        self.dataframe = None

    def input_2d_array(self):
        dialog = InputArrayDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            array = dialog.result
            QMessageBox.information(self, "Введённый массив", f"Вы ввели:\n{array}")

    def generate_2d_array(self):
        rows, cols = 100, 2  # Указываем нужное количество строк и количество колонок
        array = np.random.rand(rows, cols)  # Генерируем массив

        # Сохранение массива в CSV файл в заданном формате
        save_path = 'generated_data.csv'  # Название файла для сохранения
        header = 'x,y'  # Заголовки столбцов

        np.savetxt(save_path, array, delimiter=',', fmt='%.5f', header=header, comments='')  # Сохраняем в CSV
        QMessageBox.information(self, "Сохранение массива", f"Массив сохранен в {save_path}.")

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV files (*.csv)")
        if file_path:
            try:
                self.dataframe = DataEditor.read_csv_to_dataframe(file_path)  # Читаем CSV в DataFrame
                QMessageBox.information(self, "Информация", f"Данные загружены из {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    def build_confidence_interval(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV.")
            return

        target_column = "y"
        try:
            lower_bound, upper_bound = MathFunctions.confidence_interval(self.dataframe, target_column)
            QMessageBox.information(self, "Доверительный интервал",
                                    f"Доверительный интервал для '{target_column}': ({lower_bound}, {upper_bound})")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def check_multicollinearity(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV.")
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

    def build_linear_regression(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV.")
            return

        target_column = "y"

        try:
            coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)

            if len(self.dataframe) <= len(coefficients):
                raise ValueError("Not enough data for regression.")

            linear_func = MathFunctions.create_linear_function(coefficients)
            x_values = self.dataframe.drop(columns=[target_column]).values

            if x_values.shape[1] > 1:
                regression_line = np.array([linear_func(*row) for row in x_values])
            else:
                regression_line = np.array([linear_func(x) for x in x_values.flatten()])

            self.figure1.plot_data(x_values.flatten(), self.dataframe[target_column].values, regression_line)

            r_squared = MathFunctions.calculate_r_squared(self.dataframe, target_column, regression_line)
            QMessageBox.information(self, "Результат регрессии",
                                    f"Коэффициенты: {coefficients}, R^2: {r_squared}")

        except (ValueError, KeyError, Exception) as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении регрессии: {e}")

    def calculate_rmse(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV.")
            return

        target_column = "y"
        try:
            coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)
            linear_func = MathFunctions.create_linear_function(coefficients)
            rmse_value = MathFunctions.calculate_rmse(self.dataframe, linear_func, target_column)
            QMessageBox.information(self, "Результат", f"Среднеквадратичное отклонение (СКО): {rmse_value:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def calculate_r_squared(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные из CSV.")
            return

        target_column = "y"
        try:
            coefficients = MathFunctions.calculate_mnk_coefficients(self.dataframe, target_column)
            linear_func = MathFunctions.create_linear_function(coefficients[::-1])
            x_values = self.dataframe.drop(columns=[target_column]).values
            regression_line = np.array([linear_func(*row) for row in x_values])
            r_squared_value = MathFunctions.calculate_r_squared(self.dataframe, target_column, regression_line)
            QMessageBox.information(self, "Результат", f"Коэффициент детерминации (R²): {r_squared_value:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def update_graph1(self):
        # Логика обновления графика 1
        pass

    def update_graph2(self):
        # Логика обновления графика 2
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())