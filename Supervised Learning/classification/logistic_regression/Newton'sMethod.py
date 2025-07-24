import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def logistic(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    predictions = logistic(z)
    epsilon = 1e-5  # to avoid log(0)
    cost = (-1 / m) * (y @ np.log(predictions + epsilon) + (1 - y) @ np.log(1 - predictions + epsilon))
    return cost


def newtons_method(X, y, theta, iterations=10):
    m, n = X.shape
    cost_history = []

    for i in range(iterations):
        h = logistic(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        R = np.diag((h * (1 - h)).flatten())  
        H = (1/m) * X.T @ R @ X
        theta = theta - np.linalg.inv(H) @ gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        print(f"Iteration {i+1}: Cost {cost}")

    return theta, cost_history


def predict(X, theta):
    probs = logistic(X @ theta)
    return (probs >= 0.5).astype(int)


df = pd.read_csv("Raisin_Dataset.csv")
X = df[["Area","MajorAxisLength","MinorAxisLength","Eccentricity","ConvexArea","Extent","Perimeter"]].values  
y = df["Class"].map({"Besni": 0, "Kecimen": 1}).values
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
theta_initial = np.zeros(X_with_intercept.shape[1])

theta_final, cost_history = newtons_method(X_with_intercept, y, theta_initial)

print("Learned parameters:", theta_final)

predictions = predict(X_with_intercept, theta_final)

accuracy = np.mean(predictions == y)
print("Training accuracy:", accuracy)