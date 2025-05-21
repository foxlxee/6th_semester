import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd

def main():
    # Используем датасет диабета Pima Indian из открытого источника
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)
    X = data.drop(columns=['Outcome']).values
    y = data['Outcome'].values

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Задача 1.1
    depths = range(1, 21)
    accuracies_depth = []

    for depth in depths:
        rf = RandomForestClassifier(max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies_depth.append(acc)

    plt.figure(figsize=(8,5))
    plt.plot(depths, accuracies_depth, marker='o')
    plt.title('Точность случайного леса в зависимости от глубины деревьев')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("rf_accuracy_vs_depth.png")
    plt.close()

    # Задача 1.2
    features_options = ['auto', 'sqrt', 'log2', 1, 2, 3, 4, 5, 6, 7, 8]
    accuracies_features = []

    for max_feat in features_options:
        rf = RandomForestClassifier(max_features=max_feat, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies_features.append(acc)

    plt.figure(figsize=(8,5))
    plt.plot([str(f) for f in features_options], accuracies_features, marker='o')
    plt.title('Точность случайного леса в зависимости от max_features')
    plt.xlabel('max_features')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("rf_accuracy_vs_max_features.png")
    plt.close()

    # Задача 1.3
    estimators = range(1, 101, 10)
    accuracies_estimators = []
    times_estimators = []

    for n in estimators:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        start = time()
        rf.fit(X_train, y_train)
        end = time()
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        accuracies_estimators.append(acc)
        times_estimators.append(end - start)

    fig, ax1 = plt.subplots(figsize=(10,6))

    color = 'tab:blue'
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(estimators, accuracies_estimators, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Training time (sec)', color=color)
    ax2.plot(estimators, times_estimators, marker='x', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Точность и время обучения случайного леса в зависимости от числа деревьев')
    plt.savefig("rf_accuracy_and_time_vs_estimators.png")
    plt.close()

    # Задача 2
    # Базовая модель XGBoost
    xgb_base = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42)

    start = time()
    xgb_base.fit(X_train, y_train)
    xgb_base_time = time() - start

    preds_base = xgb_base.predict(X_test)
    xgb_base_acc = accuracy_score(y_test, preds_base)

    print(f"XGBoost базовый: Accuracy = {xgb_base_acc:.4f}, Время обучения = {xgb_base_time:.4f} сек")

    # Подбор гиперпараметров вручную для улучшения модели
    xgb_tuned = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42)

    start = time()
    xgb_tuned.fit(X_train, y_train)
    xgb_tuned_time = time() - start

    preds_tuned = xgb_tuned.predict(X_test)
    xgb_tuned_acc = accuracy_score(y_test, preds_tuned)

    print(f"XGBoost с подбором гиперпараметров: Accuracy = {xgb_tuned_acc:.4f}, Время обучения = {xgb_tuned_time:.4f} сек")

if __name__ == "__main__":
    main()
