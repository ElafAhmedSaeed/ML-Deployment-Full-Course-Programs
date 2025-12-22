# Activity 4: Compare how Pickle and Joblib save and load a trained model — in terms of file size and speed.

# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib, pickle, time, os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# --- Save using Pickle ---
t1 = time.time()
with open("lr_pickle.pkl", "wb") as f:
    pickle.dump(model, f)
pickle_time = time.time() - t1
pickle_size = os.path.getsize("lr_pickle.pkl")

# --- Save using Joblib ---
t2 = time.time()
joblib.dump(model, "lr_joblib.joblib")
joblib_time = time.time() - t2
joblib_size = os.path.getsize("lr_joblib.joblib")

# --- Compare results ---
print("Pickle save time: {:.5f}s | File size: {} bytes".format(pickle_time, pickle_size))
print("Joblib save time: {:.5f}s | File size: {} bytes".format(joblib_time, joblib_size))

# --- Load both models to check ---
t3 = time.time()
with open("lr_pickle.pkl", "rb") as f:
    loaded_pickle = pickle.load(f)
print("Pickle load time: {:.5f}s".format(time.time() - t3))

t4 = time.time()
loaded_joblib = joblib.load("lr_joblib.joblib")
print("Joblib load time: {:.5f}s".format(time.time() - t4))



################################
#Hint Code

from sklearn.linear_model import LogisticRegression
import joblib, pickle, time

model = LogisticRegression()
model.fit(X, y)

# Save using pickle
t1 = time.time()
pickle.dump(model, open("lr_pickle.pkl", "wb"))
print("Pickle time:", time.time()-t1)

# Save using joblib
t2 = time.time()
joblib.dump(model, "lr_joblib.joblib")
print("Joblib time:", time.time()-t2)

