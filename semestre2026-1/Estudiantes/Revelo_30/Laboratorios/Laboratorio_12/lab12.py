import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the digits dataset
digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)

# Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(digits.data[i].reshape(8, 8), cmap="gray")
    axes[i].set_title(f"Label: {digits.target[i]}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# 2. Random forest classifier
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
)

# Scale the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 3. Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(X_test_scaled[i].reshape(8, 8), cmap="gray")
    axes[i].set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
