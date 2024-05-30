import numpy as np
import matplotlib.pyplot as plt


def logistic_map(r, x):
    """
    Функция логистической карты

    Параметры:
    r: параметр управления логистической карты
    x: текущее значение

    Возвращает следующее значение логистической карты
    """
    return r * x * (1 - x)


def lyapunov_exponent(r, iterations):
    """
    Функция для вычисления показателя Ляпунова

    Параметры:
    r: массив значений параметра для логистической карты
    iterations: количество итераций для вычисления показателя Ляпунова

    Выводит график бифуркационной диаграммы и показатель Ляпунова для каждого значения r
    """

    last = 150

    # Инициализация x
    x = 1e-5 * np.ones(len(r))

    # Инициализация показателя Ляпунова
    lyapunov = np.zeros(len(r))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    # Вычисление показателя Ляпунова
    for i in range(iterations):
        x = logistic_map(r, x)
        lyapunov += np.log(abs(r - 2 * r * x))
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', alpha=1.0)

    ax1.set_xlim(2.5, 4)
    ax1.set_title("Бифуркационная диаграмма")

    lyapunov = lyapunov / iterations
    plt.plot(r, lyapunov, color='red', alpha=0.5)
    ax2.plot(r, lyapunov, '.g', alpha=0.5, ms=.5)
    ax2.set_xlim(2.5, 4)
    ax2.set_ylim(-4, 1)
    ax2.set_title("Показатель Ляпунова")
    plt.tight_layout()
    plt.show()


# Задаем диапазон значений r и количество итераций
r = np.linspace(2.5, 4.0, 1000)
iterations = 10000
lyapunov_exponent(r, iterations)
