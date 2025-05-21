import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_recall_curve, roc_curve
)

# Загрузка и масштабирование данных
def load_and_prepare_data():
    df = pd.read_csv('diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42), df.columns[:-1]

# Вывод стандартных метрик классификации
def print_metrics(y_true, y_pred, model_name):
    print(f'\n=== {model_name} ===')
    print(classification_report(y_true, y_pred, digits=4))
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 Score:', f1_score(y_true, y_pred))

# Оценка влияния глубины дерева на F1-метрику
def evaluate_depth(X_train, X_test, y_train, y_test):
    depths = range(1, 21)
    f1_scores = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))

    optimal_depth = depths[np.argmax(f1_scores)]
    print(f"\nOptimal depth: {optimal_depth}, Max F1 Score: {max(f1_scores):.4f}")

    # Построение графика F1 от глубины
    plt.figure(figsize=(10,6))
    plt.plot(depths, f1_scores, marker='o')
    plt.title('F1 Score vs Tree Depth')
    plt.xlabel('Tree Depth')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig('f1_vs_depth.png')
    plt.show()

    return optimal_depth

# Визуализация дерева решений с помощью Graphviz
def visualize_tree(model, feature_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=['No Diabetes', 'Diabetes'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("diabetes_tree")
    print("Дерево сохранено как diabetes_tree.pdf")

# Построение графика важности признаков
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()

# Построение PR-кривой и ROC-кривой
def plot_pr_roc_curves(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    plt.figure(figsize=(12,5))

    # PR-кривая
    plt.subplot(1,2,1)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()

    # ROC-кривая
    plt.subplot(1,2,2)
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()

    plt.tight_layout()
    plt.savefig('pr_roc_curves.png')
    plt.show()

# Дополнительно: исследование влияния max_features на F1
def analyze_max_features(X_train, X_test, y_train, y_test, depth, n_features):
    scores = []
    values = range(1, n_features + 1)
    for val in values:
        model = DecisionTreeClassifier(max_depth=depth, max_features=val, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(f1_score(y_test, y_pred))

    plt.figure(figsize=(10,6))
    plt.plot(values, scores, marker='s')
    plt.title('F1 Score vs max_features')
    plt.xlabel('max_features')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig('f1_vs_max_features.png')
    plt.show()

# Главная функция программы
def main():
    # Загрузка и подготовка данных
    (X_train, X_test, y_train, y_test), feature_names = load_and_prepare_data()

    # Модель логистической регрессии
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print_metrics(y_test, y_pred_lr, 'Logistic Regression')

    # Модель решающего дерева с параметрами по умолчанию
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print_metrics(y_test, y_pred_dt, 'Decision Tree (default)')

    # Определение оптимальной глубины дерева
    optimal_depth = evaluate_depth(X_train, X_test, y_train, y_test)

    # Обучение дерева с оптимальной глубиной
    best_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
    best_tree.fit(X_train, y_train)

    # Визуализация дерева
    visualize_tree(best_tree, feature_names)

    # Визуализация важности признаков
    plot_feature_importances(best_tree, feature_names)

    # Построение PR и ROC кривых
    plot_pr_roc_curves(best_tree, X_test, y_test)

    # Влияние max_features
    analyze_max_features(X_train, X_test, y_train, y_test, optimal_depth, len(feature_names))

if __name__ == "__main__":
    main()
