import numpy as np
import matplotlib.pyplot as plt

X = np.array([5,10,15,20,25,30])
Y = np.array([10,18,25,33,40,48])

m = 0
b = 0
learning_rate = 0.001
epochs = 1000

n = len(X)

for i in range(epochs):
    Y_pred = m*X + b
    Dm = (-2/n) * sum(X*(Y - Y_pred))
    Db = (-2/n) * sum(Y - Y_pred)

    m = m - learning_rate * Dm
    b = b - learning_rate * Db

print("Slope:", m)
print("Intercept:", b)

plt.scatter(X, Y)
plt.plot(X, m*X + b, color='red')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
