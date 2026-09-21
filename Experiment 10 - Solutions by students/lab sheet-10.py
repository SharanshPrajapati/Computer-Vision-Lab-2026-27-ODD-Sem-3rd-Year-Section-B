import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------------------------------
# Output Directory Setup
# -------------------------------------------------------------------------
OUTPUT_DIR = "experiment_10_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved automatically to: ./{OUTPUT_DIR}/")

# -------------------------------------------------------------------------
# Step 1 & 2: Load Dataset (MNIST Handwritten Digits)
# -------------------------------------------------------------------------
print("Loading MNIST dataset...")
(X_train_raw, y_train_raw), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# -------------------------------------------------------------------------
# Step 3: Image Preprocessing (Reshape, Normalize, Split)
# -------------------------------------------------------------------------
X_train_raw = np.expand_dims(X_train_raw, axis=-1).astype("float32") / 255.0
X_test = np.expand_dims(X_test, axis=-1).astype("float32") / 255.0

val_split = 10000
X_val = X_train_raw[:val_split]
y_val = y_train_raw[:val_split]
X_train = X_train_raw[val_split:]
y_train = y_train_raw[val_split:]

# -------------------------------------------------------------------------
# Step 4 & 5: Model Design and Compilation
# -------------------------------------------------------------------------
model = models.Sequential(
    [
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------------
# Step 6: Train Model & Automatically Save Learning Curves
# -------------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1,
)

# Plot & Save Accuracy/Loss Curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Val Loss", marker="o")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "1_training_curves.png")
plt.savefig(curve_path, dpi=300)
plt.close()
print(f"Saved learning curves to: {curve_path}")

# -------------------------------------------------------------------------
# Step 7: Model Evaluation & Automatically Save Confusion Matrix
# -------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save Text Classification Report to File
report = classification_report(y_test, y_pred, digits=4)
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Saved text report to: {report_path}")

# Plot & Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix on Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

cm_path = os.path.join(OUTPUT_DIR, "2_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix to: {cm_path}")

# -------------------------------------------------------------------------
# Step 8 & 9: Automatically Save Sample Predictions
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    pred_label = y_pred[i]
    true_label = y_test[i]
    color = "green" if pred_label == true_label else "red"
    plt.title(f"Pred: {pred_label} | True: {true_label}", color=color)
    plt.axis("off")

plt.suptitle(
    "Sample Predictions on Test Set (Green: Correct, Red: Error)", fontsize=14
)
plt.tight_layout()

pred_path = os.path.join(OUTPUT_DIR, "3_sample_predictions.png")
plt.savefig(pred_path, dpi=300)
plt.close()
print(f"Saved sample predictions to: {pred_path}")

print(f"\nAll task images and reports have been saved to '{OUTPUT_DIR}/'.")