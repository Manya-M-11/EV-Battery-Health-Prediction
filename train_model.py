import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy training data (speed, distance -> battery usage)
X = np.array([
    [40, 10],
    [60, 20],
    [80, 30],
    [50, 25],
    [70, 35]
])
y = np.array([5, 15, 25, 18, 30])  # battery consumption %

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")
