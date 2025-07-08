import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1/(2*m)) * np.dot(errors.T, errors)
    return cost

def batch_gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history

def predict(X, theta):
    predictions = np.dot(X, theta)
    return predictions

df = pd.read_csv("Salary_dataset.csv")
X = df[["YearsExperience"]].values  
y = df["Salary"].values     
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
theta_initial = np.zeros(2)

theta_final, cost_history = batch_gradient_descent(X_with_intercept, y, theta_initial, learning_rate=0.01, iterations=1000)

print("Learned parameters:", theta_final)

X_test = np.array([[2], [4]])
X_test_with_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]

predictions = predict(X_test_with_intercept, theta_final)
print("Predictions:", predictions)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predict(X_with_intercept, theta_final), color='red', label='Regression Line')
plt.scatter(X_test, predictions, color='green', marker='x', label='Predictions')
plt.title('Linear Regression using Batch Gradient Descent')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.grid()
plt.savefig("batch_linear_regression.png")
