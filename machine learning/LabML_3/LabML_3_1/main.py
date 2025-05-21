from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка и подготовка данных
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Названия сортов:", iris.target_names)

# Визуализация признаков
colors = ['red', 'green', 'blue']
labels = iris.target_names

# Sepal length vs Sepal width
for i, label in enumerate(labels):
    plt.scatter(df[df.target == i]['sepal length (cm)'],
                df[df.target == i]['sepal width (cm)'],
                label=label, color=colors[i])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.title('Sepal length vs Sepal width')
plt.show()

# Petal length vs Petal width
for i, label in enumerate(labels):
    plt.scatter(df[df.target == i]['petal length (cm)'],
                df[df.target == i]['petal width (cm)'],
                label=label, color=colors[i])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.title('Petal length vs Petal width')
plt.show()

# Pairplot с Seaborn
sns.pairplot(df, hue='target', palette=colors, diag_kind='kde')
plt.show()

# Подготовка бинарных датасетов
df_0_1 = df[df['target'].isin([0, 1])]  # setosa vs versicolor
df_1_2 = df[df['target'].isin([1, 2])]  # versicolor vs virginica

# Логистическая регрессия для двух датасетов ===
def train_and_evaluate(X, y, description=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({description}): {acc:.2f}")

# Setosa vs Versicolor
X_0_1 = df_0_1.drop(columns='target')
y_0_1 = df_0_1['target']
train_and_evaluate(X_0_1, y_0_1, "setosa vs versicolor")

# Versicolor vs Virginica
X_1_2 = df_1_2.drop(columns='target')
y_1_2 = df_1_2['target']
train_and_evaluate(X_1_2, y_1_2, "versicolor vs virginica")

# Генерация случайного датасета и бинарная классификация
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1, n_clusters_per_class=1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Сгенерированные данные')
plt.show()

train_and_evaluate(X, y, "сгенерированный датасет")
