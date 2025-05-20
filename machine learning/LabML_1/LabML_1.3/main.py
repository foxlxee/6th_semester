import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

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

    # Метрики для sklearn модели
    mae_sklearn = mean_absolute_error(y_test, y_pred)
    r2_sklearn = r2_score(y_test, y_pred)
    mape_sklearn = mean_absolute_percentage_error(y_test, y_pred)

    # Метрики для ручной модели
    mae_manual = mean_absolute_error(y_test, y_manual_pred)
    r2_manual = r2_score(y_test, y_manual_pred)
    mape_manual = mean_absolute_percentage_error(y_test, y_manual_pred)

    print("\nМетрики для модели Scikit-Learn:")
    print(f"MAE: {mae_sklearn:.2f}")
    print(f"R2: {r2_sklearn:.2f}")
    print(f"MAPE: {mape_sklearn:.2f}%")

    print("\nМетрики для ручной модели:")
    print(f"MAE: {mae_manual:.2f}")
    print(f"R2: {r2_manual:.2f}")
    print(f"MAPE: {mape_manual:.2f}%")

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

if __name__ == "__main__":
    main()
