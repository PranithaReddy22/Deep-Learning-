import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=10)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Predict output
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Plot data points
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')

# Decision boundary
coef = model.coef_[0]
intercept = model.intercept_

x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(coef[0]*x_vals + intercept)/coef[1]

plt.plot(x_vals, y_vals, 'k--')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Separability Demonstration")
plt.show()
