import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

x = np.arange(-10,10,0.2)
y = sigmoid(x)

plt.plot(x,y,color='green')
plt.title("Sigmoid Function for Logistic Regression")
plt.xlabel("Input values")
plt.ylabel("Sigmoid Output")
plt.grid(True)
plt.show()
