import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Генерация чисел по равномерному распределению
def generate_uniform(size, low=0, high=1):
    uniform_numbers = np.random.uniform(low, high, size)
    return uniform_numbers

# 2. Генерация чисел по нормальному распределению
def generate_normal(size, mean=0, std_dev=1):
    normal_numbers = np.random.normal(mean, std_dev, size)
    return normal_numbers

# 3. Генерация чисел по распределению закона Пирса (распределение Бета)
def generate_beta(size, a=2, b=5):
    beta_numbers = np.random.beta(a, b, size)
    return beta_numbers

# 4. Генерация чисел по распределению Фишера (распределение F)
def generate_fisher(size, dfn=5, dfd=2):
    fisher_numbers = np.random.f(dfn, dfd, size)
    return fisher_numbers

# Пример использования функций
size = 1000

uniform_data = generate_uniform(size)
normal_data = generate_normal(size)
beta_data = generate_beta(size)
fisher_data = generate_fisher(size)

# Визуализация результатов
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.hist(uniform_data, bins=30, alpha=0.7, color='blue')
plt.title('Равномерное распределение')

plt.subplot(2, 2, 2)
plt.hist(normal_data, bins=30, alpha=0.7, color='green')
plt.title('Нормальное распределение')

plt.subplot(2, 2, 3)
plt.hist(beta_data, bins=30, alpha=0.7, color='orange')
plt.title('Распределение закона Пирса (Бета)')

plt.subplot(2, 2, 4)
plt.hist(fisher_data, bins=30, alpha=0.7, color='red')
plt.title('Распределение Фишера')

plt.tight_layout()
plt.show()