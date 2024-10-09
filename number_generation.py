import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class GeneratingNumbers (object):
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
        #запись данных в файл, через ; с двумя цифрами после запятой, позже можно будет исправить на n-ое кол-во после запятой
        np.savetxt("random_data.csv", numbers, delimiter=";", fmt="%d")

    #можем ли мы сделать метод визуализации (?) - однозначно!
    def visualisation(self, numbers):
        plt.subplot(2, 2, 1)
        plt.hist(numbers, bins=30, alpha=0.7, color='blue')
        plt.show()
    

distribution_options = int(input ("select the distribution option, 1 - uniform, 2 - normal, 3 - Pierce's law, 4 - Fisher's law  "))

amount_of_numbers = int(input("set the number of generated numbers  "))

print("write a range of numbers   ")

#для последних двух действует ограничение: они > 0 
parametr1 = int(input())
parametr2 = int(input())

obj = GeneratingNumbers()

#сохраним массив для визуализации
our_data = obj.option_recognition(distribution_options, amount_of_numbers, parametr1, parametr2)

#сохраняем его в csv
obj.data_save(our_data)

# Визуализация результатов
obj.visualisation(our_data)