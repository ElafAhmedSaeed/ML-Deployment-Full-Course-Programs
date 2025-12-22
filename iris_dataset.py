from sklearn.datasets import load_iris
data = load_iris()

X = data.data        # features (sepal/petal)
y = data.target      # labels (0, 1, 2)

print(X)
print("y = ", y)



