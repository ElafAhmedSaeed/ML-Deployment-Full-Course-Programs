
# 1. Import required libraries
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 2. Load dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Save the model as tree_model.pkl
with open("tree_model.pkl", "wb") as file:
    pickle.dump(model, file)

# 6. Load the saved model
with open("tree_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# 7. Predict on a new sample (e.g., one flower)
sample = [[5.1, 3.5, 1.4, 0.2]]   # Sepal & Petal measurements
prediction = loaded_model.predict(sample)

# 8. Display result
print("Predicted class:", iris.target_names[prediction][0])

