import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -----------------------------
# 1) Create a simple dataset
#    Example: y = 2x + 1 + noise
# -----------------------------
np.random.seed(42)

X = np.random.rand(500, 1).astype(np.float32)         # shape: (500, 1)
noise = (np.random.randn(500, 1) * 0.05).astype(np.float32)
y = (2 * X + 1 + noise).astype(np.float32)            # regression target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2) Define and train the model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Optional: evaluate
print("Test MSE:", model.evaluate(X_test, y_test, verbose=0))

# -----------------------------
# 3) Save and reload model (SavedModel format)
# -----------------------------
save_path = "my_saved_model"
model.save(save_path)  # creates a folder

new_model = tf.keras.models.load_model(save_path)

# -----------------------------
# 4) Predict on test data
# -----------------------------
preds = new_model.predict(X_test[:5])
print("X_test[:5] =", X_test[:5].reshape(-1))
print("Predictions =", preds.reshape(-1))
print("True y_test[:5] =", y_test[:5].reshape(-1))
