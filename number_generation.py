import numpy as np
import matplotlib.pyplot as plt
import csv


class GeneratingNumbers (object):

    #метод создания массива как объекта
    def __init__(self, distribution_options, amount_of_numbers, parametr1, parametr2):
        self.distribution_options = distribution_options
        self.amount_of_numbers = amount_of_numbers
        self.parametr1 = parametr1
        self.parametr2 = parametr2

    def option_recognition(self, distribution_options, amount_of_numbers, parametr1, parametr2):
        # 1. Генерация чисел по равномерному распределению
        if distribution_options == 1:
            numbers = np.random.uniform(parametr1, parametr2, amount_of_numbers)

        # 2. Генерация чисел по нормальному распределению
        if distribution_options == 2:
            numbers = np.random.normal(parametr1, parametr2, amount_of_numbers)

        # 3. Генерация чисел по распределению закона Пирса (распределение Бета)
        if distribution_options == 3:
            numbers = np.random.beta(parametr1, parametr2, amount_of_numbers)

        # 4. Генерация чисел по распределению Фишера (распределение F)
        if distribution_options == 4:
            numbers = np.random.f(parametr1, parametr2, amount_of_numbers)
        return numbers
    
    def data_save(self, numbers):
        with open("random_data_array_1d", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(numbers)


    """#можем ли мы сделать метод визуализации (?) - однозначно!
    def visualisation(self, numbers):
        plt.subplot(2, 2, 1)
        plt.hist(numbers, bins=30, alpha=0.7, color='blue')
        plt.show()"""


class Array2DGeneration(GeneratingNumbers):
    def __init__(self, distribution_options, amount_of_rows, amount_of_columns, parametr1, parametr2):
        super().__init__(distribution_options, amount_of_rows * amount_of_columns, parametr1, parametr2)
        self.amount_of_rows = amount_of_rows
        self.amount_of_columns = amount_of_columns

    def generate_2d_array(self):
        # Генерация двумерного массива
        numbers = self.option_recognition(self.distribution_options, self.amount_of_numbers, self.parametr1, self.parametr2)
        return numbers.reshape((self.amount_of_rows, self.amount_of_columns))

    def data_save(self, numbers):
        with open("random_data_array_2d.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(numbers)


if __name__ == "__main__":
    # Генерация двумерного массива с нормальным распределением
    generator = Array2DGeneration(distribution_options=2, amount_of_rows=5, amount_of_columns=4, parametr1=0, parametr2=1)
    array_2d = generator.generate_2d_array()
    print(array_2d)
    generator.data_save(array_2d)