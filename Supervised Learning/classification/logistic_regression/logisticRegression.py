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

def batch_gradient_ascent(X, y, theta, learning_rate=0.1, iterations=1000):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        z = X.dot(theta)
        predictions = logistic(z)
        errors = y - predictions
        gradient = (1/m) * X.T.dot(errors)
        theta = theta + learning_rate * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history

def predict(X, theta):
    z = X.dot(theta)
    predictions = logistic(z)
    return predictions

def classify(predictions, threshold=0.5):
    return (predictions >= threshold).astype(int)

df = pd.read_csv("Raisin_Dataset.csv")
X = df[["Eccentricity","MajorAxisLength"]].values  
y = df["Class"].map({"Besni": 0, "Kecimen": 1}).values
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
theta_initial = np.zeros(X_with_intercept.shape[1]) 


theta_final, cost_history = batch_gradient_ascent(X_with_intercept, y, theta_initial, learning_rate=0.2, iterations=20000)

print("Learned parameters:", theta_final)

probs = predict(X_with_intercept, theta_final)
predictions = classify(probs)


accuracy = np.mean(predictions == y)
print("Training accuracy:", accuracy)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.xlabel("Area")
plt.ylabel("Major Axis Length")
plt.title("2D Feature Space Colored by Class")
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(theta_final[0] + theta_final[1] * x_vals) / theta_final[2]
plt.plot(x_vals, y_vals, color="black", label="Decision Boundary")
plt.legend()
plt.grid(True)
plt.savefig("logistic_regression.png")

