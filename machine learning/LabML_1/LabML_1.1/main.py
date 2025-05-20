import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Функция чтения данных из CSV-файла
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Функция вывода статистической информации о данных
def show_statistics(df):
    print("\nСтатистика по данным:")
    print("Количество значений:\n", df.count())
    print("Минимум:\n", df.min())
    print("Максимум:\n", df.max())
    print("Среднее:\n", df.mean())

# Построение графика исходных данных
def plot_original_data(x, y, title='Исходные данные'):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    return plt

# Реализация метода наименьших квадратов
def least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    b0 = y_mean - b1 * x_mean
    return b0, b1

# Построение регрессионной прямой на графике
def plot_regression_line(x, y, b0, b1, title='Линейная регрессия'):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue', label='Данные')
    y_pred = b0 + b1 * x
    plt.plot(x, y_pred, color='red', label='Регрессионная прямая')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    return plt

# Отображение квадратов ошибок на графике
def plot_error_squares(x, y, b0, b1, title='Квадраты ошибок'):
    y_pred = b0 + b1 * x
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue', label='Данные')
    plt.plot(x, y_pred, color='red', label='Регрессионная прямая')

    # Отрисовка прямоугольников (квадратов ошибок)
    for xi, yi, ypi in zip(x, y, y_pred):
        plt.plot([xi, xi], [yi, ypi], color='green', linestyle='--')  # пунктирные линии
        plt.fill_between([xi - 0.1, xi + 0.1], yi, ypi, color='green', alpha=0.3)  # прямоугольники

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    return plt

# Основная функция
def main():
    # Ввод пути к CSV-файлу
    file_path = input("Введите путь к CSV-файлу: ")
    df = read_data(file_path)

    # Выводим список столбцов и предлагаем выбрать, какие будут X и Y
    print("\nСтолбцы в файле:", list(df.columns))
    x_col = input("Выберите столбец для X: ")
    y_col = input("Выберите столбец для Y: ")

    x = df[x_col].values
    y = df[y_col].values

    # Выводим статистику по выбранным столбцам
    show_statistics(df[[x_col, y_col]])

    # Построение графиков
    plot1 = plot_original_data(x, y, 'Исходные данные')
    b0, b1 = least_squares(x, y)
    plot2 = plot_regression_line(x, y, b0, b1, 'Линейная регрессия')
    plot3 = plot_error_squares(x, y, b0, b1, 'Квадраты ошибок')

    # Отображение всех трёх графиков
    plot1.show()
    plot2.show()
    plot3.show()

# Точка входа в программу
if __name__ == "__main__":
    main()
