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


class ArrayGeneration(GeneratingNumbers):
    def __init__(self, distribution_options, size1, size2, parametr1, parametr2):
        self.distribution_options = distribution_options
        self.size1 = size1
        self.size2 = size2
        self.parametr1 = parametr1
        self.parametr2 = parametr2

    #создаю двумерный массив и сохраняю, используя наследственный метод
    def creating(self, distribution_options, size1, size2, parametr1, parametr2):
        arraysize = size1*size2
        array1d = super().option_recognition(distribution_options, arraysize, parametr1, parametr2)
        array2d = np.reshape(array1d, (size1, size2))
        return array2d
    
    #сохранение двумерного массива в csv
    def data_save(self, array2d):
        with open("random_data_array_2d", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(array2d)
        print("Successfully saved")

    
#просим на вход вариант распределения, кол-во чисел
distribution_options = int(input ("select the distribution option, 1 - uniform, 2 - normal, 3 - Pierce's law, 4 - Fisher's law  "))

amount_of_numbers = int(input("set the number of generated numbers  "))

print("write parametrs for generating of numbers   ")

#для последних двух действует ограничение: они > 0 
parametr1 = int(input())
parametr2 = int(input())

if distribution_options == 3 or distribution_options == 4:
    if parametr1 > 0 and parametr2 > 0:
        pass
    else:
        print("invalid values, please specify positive values")
        
#задание размера массива
size1 = int(input("введите предполагаемый размер массива    "))
size2 = int(input())

amount_of_numbers = size1*size2

obj = GeneratingNumbers(distribution_options, amount_of_numbers, parametr1, parametr2)

#сохраним массив для визуализации
our_data1d = obj.option_recognition(distribution_options, amount_of_numbers, parametr1, parametr2)

#сохраняем его в csv
obj.data_save(our_data1d)

"""# Визуализация результатов
obj.visualisation(our_data1d)"""


array2d = ArrayGeneration(distribution_options, size1, size2, parametr1, parametr2)

#сохраним массив
our_data2d = array2d.creating(distribution_options, size1, size2, parametr1, parametr2)

array2d.data_save(our_data2d)