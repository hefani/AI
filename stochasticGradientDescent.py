import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1/(2*m)) * np.dot(errors.T, errors)
    return cost


def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, epochs=50):
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index+1]
            yi = y[rand_index:rand_index+1]

            prediction = xi.dot(theta)
            error = prediction - yi
            gradient = xi.T.dot(error)
            theta = theta - learning_rate * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        print(f"Epoch {epoch+1}: Cost {cost}")

    return theta, cost_history


def predict(X, theta):
    predictions = np.dot(X, theta)
    return predictions


df = pd.read_csv("Salary_dataset.csv")
X = df[["YearsExperience"]].values  
y = df["Salary"].values     


X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]


theta_initial = np.zeros(2)

# Run SGD
theta_final, cost_history = stochastic_gradient_descent(X_with_intercept, y, theta_initial, learning_rate=0.01, epochs=50)
predictions = predict(X_with_intercept, theta_final)

print("Learned parameters (theta):", theta_final)


plt.scatter(X, y, color='blue', label='Data Points') 
plt.plot(X, predict(X_with_intercept, theta_final), color='red', label='Regression Line')  
plt.title('Linear Regression using Stochastic Gradient Descent')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.grid()
plt.savefig("stochatic_linear_regression.png")