import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel, \
    QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from generate_data_csv import generate_and_save_data

class MyFigure(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Тестовый график')
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
        self.setWindowTitle('Графики и загрузка CSV')
        self.setGeometry(100, 100, 1800, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.animation_fig = MyFigure()
        self.layout.addWidget(self.animation_fig)

        self.csv_fig = MyFigure()
        self.layout.addWidget(self.csv_fig)

        self.button = QPushButton('Открыть CSV файл')
        self.button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.button)

        self.generate_button = QPushButton('Сгенерировать данные')
        self.generate_button.clicked.connect(self.generate_data_and_plot)
        self.layout.addWidget(self.generate_button)

        self.start_animation()

    def start_animation(self):
        self.x = np.linspace(0, 2 * np.pi, 100)
        self.angle = 0
        self.animation_fig.line1, = self.animation_fig.ax.plot(self.x, np.sin(self.x))
        self.animation_fig.line2, = self.animation_fig.ax.plot(self.x, np.tan(self.x))
        self.animation_anim = FuncAnimation(self.animation_fig.figure, self.update_animation, frames=100, interval=50)

    def update_animation(self, frame):
        self.angle += 0.1
        self.animation_fig.line1.set_ydata(np.sin(self.x + self.angle))
        self.animation_fig.line2.set_ydata(np.tan(self.x + self.angle))
        return self.animation_fig.line1, self.animation_fig.line2

    def load_csv(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть CSV файл', '', 'CSV files (*.csv);;All files (*)')
            if file_name:
                data = pd.read_csv(file_name)
                x = data['x']
                y = data['y']
                self.csv_fig.plot_data(x, y)
                self.plot_regression(x, y)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Ошибка при загрузке файла: {e}')

    def plot_regression(self, x, y):
        model = LinearRegression()
        x_reshaped = x.values.reshape(-1, 1)
        model.fit(x_reshaped, y)
        y_pred = model.predict(x_reshaped)
        self.csv_fig.plot_data(x, y, y_pred)

    def generate_data_and_plot(self):
        try:
            file_name = generate_and_save_data()
            self.load_csv(file_name)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Ошибка во время генерации данных: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())