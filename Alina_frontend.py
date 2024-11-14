import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QLabel, QHBoxLayout, QMessageBox, QLineEdit, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Liza_backend import read_csv_to_dataframe, calculate_mnk_coefficients  # Импортируем нужные функции из liza_backend
import scipy.stats as stats  # Для работы с доверительными интервалами


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
        self.distribution1_combo.addItems(["Нормальное", "Равномерное"])  # Примеры
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

        # Дополнительные кнопки
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
        self.confidence_interval_button.clicked.connect(self.build_confidence_interval)  # Обработка доверительного интервала
        button_layout.addWidget(self.confidence_interval_button)

        self.extra_button1 = QPushButton('Построить линейную регрессию')
        self.extra_button2 = QPushButton('Проверка на мультиколлинеарность')
        self.extra_button1.clicked.connect(self.build_linear_regression)  # Подключаем функцию линейной регрессии
        button_layout.addWidget(self.extra_button1)
        button_layout.addWidget(self.extra_button2)

        layout.addStretch()
        central_widget.setLayout(layout)

        self.dataframe = None  # Указываем DataFrame, который будет хранить данные

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV files (*.csv)")
        if file_path:
            try:
                self.dataframe = read_csv_to_dataframe(file_path)
                QMessageBox.information(self, "Информация", f"Данные загружены из {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    def build_linear_regression(self):
        if self.dataframe is not None:

            target_column = 'Построить линейную регрессию'
            try:
                coefficients = calculate_mnk_coefficients(self.dataframe, target_column)
                x_values = self.dataframe.drop(columns=[target_column])
                intercept = coefficients[0]
                slope = coefficients[1]
                regression_line = intercept + slope * x_values  # Расчет линейной линии регрессии
                self.figure1.plot_data(x_values, self.dataframe[target_column], regression_line)
                QMessageBox.information(self, "Данные линейной регрессии", f"Коэффициенты: {coefficients}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
        else:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные из CSV.")

    def input_2d_array(self):
        pass  # Заглушка;

    def generate_2d_array(self):
        # Генерация случайного двумерного массива
        rows, cols = 5, 5
        array = np.random.rand(rows, cols)
        QMessageBox.information(self, "Сгенерированный массив", str(array))

    def build_confidence_interval(self):
        sample_data = np.random.normal(0, 1, 100)  # Генерация выборки
        confidence = 0.95
        mean = np.mean(sample_data)
        se = stats.sem(sample_data)
        interval = stats.t.interval(confidence, len(sample_data)-1, loc=mean, scale=se)
        QMessageBox.information(self, "Доверительный интервал", f"Доверительный интервал: {interval}")

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