# student_performance_tf.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== 1. GENERATE SYNTHETIC DATA ====================
np.random.seed(42)
num_samples = 500

study_hours = np.random.randint(0, 15, num_samples)
attendance = np.random.randint(50, 101, num_samples)
past_score = np.random.randint(30, 101, num_samples)
sleep_hours = np.random.randint(4, 10, num_samples)
internet_hours = np.random.randint(0, 6, num_samples)

X = np.column_stack((study_hours, attendance, past_score, sleep_hours, internet_hours))

# Target: Pass (1) if weighted sum > threshold else Fail (0)
weights = np.array([0.4, 0.2, 0.3, 0.05, -0.05])
y_score = X @ weights
y = (y_score > 35).astype(int)

# ==================== 2. PREPROCESS ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, input_shape=(5,), activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # Binary output
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ==================== 4. TRAIN ====================
history = model.fit(
    X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.1
)

# ==================== 5. EVALUATE ====================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
