import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd

def main():
    # Загрузка данных
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:, np.newaxis, 2]  # Используем признак 'bmi'
    y = diabetes.target

    # Разделение на обучающую и тестовую выборки
    X_train = X[:-20]
    X_test = X[-20:]
    y_train = y[:-20]
    y_test = y[-20:]

    # Линейная регрессия с использованием Scikit-Learn
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Коэффициенты sklearn: ", reg.coef_)
    print("Свободный член sklearn: ", reg.intercept_)

    # Реализация линейной регрессии вручную
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y_manual_pred = X_test_b.dot(theta_best)

    print("Коэффициенты (ручной метод): ", theta_best[1])
    print("Свободный член (ручной метод): ", theta_best[0])

    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color="black", label="Фактические значения")
    plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Scikit-Learn регрессия")
    plt.plot(X_test, y_manual_pred, color="red", linestyle='dashed', label="Ручная регрессия")
    plt.xlabel("BMI")
    plt.ylabel("Прогресс заболевания")
    plt.title("Линейная регрессия на признаке BMI")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Таблица с результатами предсказаний
    results = pd.DataFrame({
        'Фактическое значение': y_test,
        'Sklearn Предсказание': y_pred,
        'Ручное Предсказание': y_manual_pred
    })

    print("\nТаблица предсказаний:")
    print(results.round(2))

# Запуск main
if __name__ == "__main__":
    main()
