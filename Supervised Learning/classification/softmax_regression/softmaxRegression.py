import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def softmax(z):  #z is theta transpose * X
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_cost(X, y, theta, k):
    m = X.shape[0]
    z = X @ theta
    probs = softmax(z)
    y_onehot = np.eye(k)[y]
    cost = -np.sum(y_onehot * np.log(probs + 1e-9)) / m
    return cost


def softmax_regression(X, y, k, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, k))
    cost_history = []

    for i in range(iterations):
        z = X @ theta
        probs = softmax(z)
        y_onehot = np.eye(k)[y]
        gradient = (1/m) * X.T @ (probs - y_onehot)
        theta -= learning_rate * gradient

        cost = compute_cost(X, y, theta, k)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return theta, cost_history


def predict(X, theta):
    probs = softmax(X @ theta)
    return np.argmax(probs, axis=1)


df = pd.read_csv("Raisin_Dataset.csv")
features = ["MinorAxisLength","Eccentricity","ConvexArea","Extent","Perimeter", "Area", "MajorAxisLength"]
X = df[features].values
class_map = {label: idx for idx, label in enumerate(df["Class"].unique())}
y = df["Class"].map(class_map).values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_with_intercept = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

k = len(class_map)  # k is number of classes

theta_final, cost_history = softmax_regression(X_with_intercept, y, k, learning_rate=0.1, iterations=3000)

y_pred = predict(X_with_intercept, theta_final)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")


"""
#draw the boundry line

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="viridis", edgecolors="k")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Feature space colored by class")

x_vals = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200)
for i in range(k):
    for j in range(i + 1, k):
        theta_diff = theta_final[:, i] - theta_final[:, j]
        if abs(theta_diff[2]) > 1e-6:
            y_vals = -(theta_diff[0] + theta_diff[1] * x_vals) / theta_diff[2]
            plt.plot(x_vals, y_vals, label=f"Boundary {i} vs {j}")

plt.legend(handles=scatter.legend_elements()[0], labels=class_map.keys())
plt.grid(True)
plt.savefig("softmax_Regression_.png")
"""
