import re
from typing import List

def extract_coefficients(*polynomials: str) -> List[float]:
    coefficients_dict = {}

    for polynomial in polynomials:
        # Находим все коэффициенты и соответствующие переменные
        matches = re.findall(r'([+-]?\d*\.?\d*)(x\d)', polynomial)

        for match in matches:
            coeff_str, variable = match
            if coeff_str == '' or coeff_str == '+':
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)

            # Сохраняем коэффициенты в словаре
            if variable in coefficients_dict:
                coefficients_dict[variable] += coeff
            else:
                coefficients_dict[variable] = coeff

    # Возвращаем коэффициенты в порядке увеличения переменной
    max_var = max(int(key[1:]) for key in coefficients_dict.keys())
    return [coefficients_dict.get(f'x{i}', 0.0) for i in range(1, max_var + 1)]

def main():
    print("Введите многочлены (для завершения ввода нажмите Enter на пустой строке):")
    polynomials = []

    while True:
        polynomial = input()
        if not polynomial:  # Если строка пустая, завершить ввод
            break
        polynomials.append(polynomial)

    # Основной процесс вывода коэффициентов
    coefficients = extract_coefficients(*polynomials)
    print("Коэффициенты:", coefficients)

if __name__ == "__main__":
    main()