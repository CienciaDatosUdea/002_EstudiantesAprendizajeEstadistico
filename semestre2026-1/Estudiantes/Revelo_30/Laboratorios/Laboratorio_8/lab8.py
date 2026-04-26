import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

iris = load_iris()


target = pd.DataFrame(iris.target, columns=["target"])

# 1. Define the first class as 0 and the other classes as 1
target_0 = pd.DataFrame()
target_0["target"] = target["target"].apply(lambda x: 0 if x == 0 else 1)

# 2. Define the second class as 0 and the other classes as 1
target_1 = pd.DataFrame()
target_1["target"] = target["target"].apply(lambda x: 0 if x == 1 else 1)

# 3. Define the third class as 0 and the other classes as 1
target_2 = pd.DataFrame()
target_2["target"] = target["target"].apply(lambda x: 0 if x == 2 else 1)


# Solution:

# 1. DESCR

print(iris.DESCR)

# 2. DataFrame construction

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 3. Rename columns

df.columns = [
    col.replace(" ", "_").replace("(", "").replace(")", "").capitalize()
    for col in df.columns
]

# 4. Analyze the DataFrame

print("DataFrame length:")
print(len(df))

print("Null count:")
print(df.isnull().sum())

print("DataFrame info:")
print(df.info())

print("DataFrame description:")
print(df.describe())

# 5. Basic statistics

print("Mean of each column:")
print(df.mean())

print("Median of each column:")
print(df.median())

print("Standard deviation of each column:")
print(df.std())

# Plotting

for column in df.columns:
    plt.figure()
    plt.hist(df[column], bins=20, edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 6. Correlation matrix
correlation_matrix = df.corr()
print("Correlation matrix:")
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# 7. Train-test split

Xs_train, Xs_test, ys_train, ys_test = [], [], [], []

targets = [target_0, target_1, target_2]

for target in targets:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(splitter.split(df, target["target"]))

    Xs_train.append(train_idx)
    Xs_test.append(test_idx)

    ys_train.append(target["target"].iloc[train_idx])
    ys_test.append(target["target"].iloc[test_idx])

iris.target_names

# 8. logistic regression

models = []
scores = []

for i in range(3):
    model = LogisticRegression()

    model.fit(df.iloc[Xs_train[i]][["Sepal_length_cm"]], ys_train[i])

    score = model.score(df.iloc[Xs_test[i]][["Sepal_length_cm"]], ys_test[i])

    models.append(model)
    scores.append(score)

# 9. Classification boundaries

X_new = pd.DataFrame(
    np.linspace(-10, 10, 1000).reshape(-1, 1), columns=["Sepal_length_cm"]
)


probs = [model.predict_proba(X_new)[:, 1] for model in models]

decision_boundaries = [
    X_new["Sepal_length_cm"][np.argmax(probs[i] >= 0.5)] for i in range(3)
]

# 10 . Plotting decision boundaries

plt.figure(figsize=(10, 6))
plt.plot(X_new, probs[0], label="Class 0 vs Rest", color="blue")
plt.axvline(
    x=decision_boundaries[0],
    color="blue",
    linestyle="--",
    label=f"Boundary Class 0: {decision_boundaries[0]:.2f}",
)
plt.title("Logistic Regression Decision Boundaries")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Probability of Class 1")
plt.legend()
plt.grid()
plt.show()

# 11. Multiclass classification

target = pd.DataFrame(iris.target, columns=["target"])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(splitter.split(df, target["target"]))

X_train = df.iloc[train_idx]
X_test = df.iloc[test_idx]

y_train = target["target"].iloc[train_idx]
y_test = target["target"].iloc[test_idx]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print(f"Multiclass Logistic Regression Accuracy: {score:.2f}")

# 12. Given X = [4.9,5.0, 1.8, 0.3], probability of belonging to each class

X_new = pd.DataFrame(np.array([[4.9, 5.0, 1.8, 0.3]]), columns=df.columns)

probs = model.predict_proba(X_new)

print("Probabilities of belonging to each class:")
for i, class_name in enumerate(iris.target_names):
    print(f"{class_name}: {probs[0][i]:.4f}")
