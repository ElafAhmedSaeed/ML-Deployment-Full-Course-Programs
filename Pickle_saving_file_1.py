import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# =====================
# Load Dataset
# =====================
data = load_iris()
X = data.data
y = data.target

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Train a model
# =====================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# =====================
# Save model
# =====================
pickle.dump(model, open("model.pkl", "wb"))

# =====================
# Load model
# =====================
loaded_model = pickle.load(open("model.pkl", "rb"))
print(loaded_model.predict(X_test))
