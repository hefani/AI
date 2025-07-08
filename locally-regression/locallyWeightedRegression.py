import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_weight(query_point, X, tau) :
    m = X.shape[0]
    weights = np.exp(-np.sum((X - query_point)**2, axis=1) / (2 * tau**2))
    return np.diag(weights)


def predict_point(query_point, X, y, tau):
    weights = get_weight(query_point, X, tau)
    XTWX = X.T @ weights @ X   
    if np.linalg.det(XTWX) == 0:
        return query_point @ np.zeros(X.shape[1])  # fallback if matrix is not invertible
    theta = np.linalg.inv(XTWX) @ X.T @ weights @ y
    return query_point @ theta


def locally_weighted_regression(X, y, tau):
    m = X.shape[0]
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = predict_point(X[i], X, y, tau)
    return y_pred


df = pd.read_csv("Salary_dataset.csv")
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

tau = 0.5

y_pred = locally_weighted_regression(X_with_intercept, y, tau)

plt.scatter(X, y, label="Data", color="blue")
plt.plot(X, y_pred, label=f"LWR (tau={tau})", color="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Locally Weighted Linear Regression")
plt.legend()
plt.grid(True)
plt.savefig("locally_weighted_Regression.png")