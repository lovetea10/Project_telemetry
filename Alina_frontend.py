import sys
import pandas as pd
from Ui_Dialog import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QFileDialog, QMessageBox

class App(QDialog, Ui_Dialog):
    def __init__(self):
        super(App, self).__init__()

        self.setWindowTitle('Project_telemetry')

        # Устанавливаем позицию и размеры окна
        self.left = 0
        self.top = 50
        self.width = 1900
        self.height = 900
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.Button()
        self.Presentashion()

        # Выводим окно на экран
        self.show()
    def Presentashion(self):
        dialog = QDialog()
        dialog.setFixedSize(300, 200)
        self.show()
    def Button(self):
        self.button = QPushButton('Открыть CSV файл', self)
        self.button.setGeometry(1500, 700, 200, 50)  # (x, y, width, height)
        self.button.clicked.connect(self.open_csv_file)

    def open_csv_file(self):
        # Открываем диалог для выбора CSV файла
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Открыть CSV файл", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)

        if fileName:
            try:
                # Читаем CSV файл с помощью pandas
                data = pd.read_csv(fileName)
                QMessageBox.information(self, "Содержимое файла", f"Файл {fileName} успешно открыт!\n\n{data.head()}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось открыть файл: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())
