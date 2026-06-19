from pprint import pprint

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Moons dataset
X_moons, y_moons = make_moons(n_samples=3000, noise=0.25, random_state=42)

# Visualize the dataset
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, marker=".")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Moons Dataset")
plt.show()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42, stratify=y_moons
)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = keras.Sequential(
    [
        layers.Input(shape=(2,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping],
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()


# Plot decision boundary
def plot_decision_boundary(model, X, y, scaler):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    probabilities = model.predict(grid_scaled, verbose=0)
    Z = probabilities.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, alpha=0.7)
    plt.contour(xx, yy, Z, levels=[0.5])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", marker=".")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()


plot_decision_boundary(model, X_test, y_test, scaler)


# Digits dataset
digits_dataset = load_digits()

X_digits = digits_dataset.data
y_digits = digits_dataset.target

X_digits.shape
y_digits.shape

# Visualize the first 10 images and their labels
plt.figure(figsize=(8, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits_dataset.images[i], cmap="gray")
    plt.title(f"Label: {y_digits[i]}")
    plt.axis("off")
plt.suptitle("First 10 Images from Digits Dataset")
plt.tight_layout()
plt.show()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42, stratify=y_digits
)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = keras.Sequential(
    [
        layers.Input(shape=(64,)),
        layers.Dense(
            128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)
        ),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=256,
    callbacks=[early_stopping],
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict
probs = model.predict(X_test_scaled)
predictions = np.argmax(probs, axis=1)

# Visualize some predictions
plt.figure(figsize=(8, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
    plt.title(f"Pred: {predictions[i]}")
    plt.axis("off")
plt.suptitle("Predicted Labels for Test Images")
plt.tight_layout()
plt.show()

# Plot training history
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion Matrix")
plt.show()

model.summary()


# test with different units
def build_model(units):
    model = keras.Sequential(
        [
            layers.Input(shape=(64,)),
            layers.Dense(
                units, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)
            ),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


units_list = [32, 64, 128, 256]
histories = []
eval_results = []

for units in units_list:
    print(f"Training model with {units} units...")
    model = build_model(units)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=500,
        batch_size=256,
        callbacks=[early_stopping],
    )
    histories.append(history)
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    results = {"units": units, "test_loss": test_loss, "test_accuracy": test_accuracy}
    eval_results.append(results)

    # Plot training history
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (units={units})")
    plt.legend()
    plt.show()

    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy (units={units})")
    plt.legend()
    plt.show()

    # Confusion matrix
    probs = model.predict(X_test_scaled)
    predictions = np.argmax(probs, axis=1)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"Confusion Matrix (units={units})")
    plt.show()

pprint(eval_results, indent=2, sort_dicts=False)
