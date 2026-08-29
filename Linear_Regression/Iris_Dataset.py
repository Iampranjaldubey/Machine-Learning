import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data[:50, 0]
y = iris.data[:50, 1]

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

X = np.hstack((x, np.ones((x.shape[0], 1))))

w = np.linalg.inv(X.T @ X) @ X.T @ y

m = w[0, 0]
c = w[1, 0]

print("m =", m)
print("c =", c)

y_pred = X @ w

plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, label="Regression Line")
plt.xlabel("Sepal length (x)")
plt.ylabel("Sepal width (y)")
plt.title("Linear Regression using Matrix Multiplication")
plt.legend()
plt.show()


