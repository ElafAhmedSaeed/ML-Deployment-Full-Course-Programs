import joblib
from sklearn.ensemble import RandomForestClassifier
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
# Train model
# =====================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# =====================
# Save and load
# =====================
joblib.dump(model, "model.joblib")
loaded_model = joblib.load("model.joblib")

print(loaded_model.score(X_test, y_test))
