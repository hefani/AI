import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression

df = pd.read_csv("test.csv")
X = df[["x"]].values
y = df["y"].values

X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression_normal_equation(X, y):
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)

    try:
        theta = np.linalg.solve(X_transpose_X, X_transpose_y)
        return theta
    except np.linalg.LinAlgError:
        return None
    

theta = linear_regression_normal_equation(X_with_intercept, y)
if theta is not None:
    print(theta)
else:
    print("we cant!")


def predict(X, theta):
    predictions = np.dot(X, theta)
    return predictions


X_test = np.array([[2], [4]])
X_test_with_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]

predictions = predict(X_test_with_intercept, theta)
print("Predictions:", predictions)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predict(X_with_intercept, theta), color='red', label='Regression Line')
plt.scatter(X_test, predictions, color='green', marker='x', label='Predictions')
plt.title('Linear Regression using Normal Equation')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.grid()
plt.savefig("normal_equation_linear_regression.png")
