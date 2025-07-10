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


df = pd.read_csv("personality_synthetic_dataset.csv")
features = ["social_energy","alone_time_preference","talkativeness","deep_reflection","group_comfort"
            , "party_liking", "listening_skill","empathy","creativity","organization","leadership","risk_taking"
            ,"public_speaking_comfort","curiosity","routine_preference","excitement_seeking","friendliness","emotional_stability","planning"
            ,"spontaneity","adventurousness","reading_habit","sports_interest","online_social_usage","travel_desire","gadget_usage","work_style_collaborative"
            ,"decision_speed","stress_handling"]
X = df[features].values
class_map = {label: idx for idx, label in enumerate(df["personality_type"].unique())}
y = df["personality_type"].map(class_map).values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_with_intercept = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

k = len(class_map)  # k is number of classes

theta_final, cost_history = softmax_regression(X_with_intercept, y, k, learning_rate=0.1, iterations=3000)

y_pred = predict(X_with_intercept, theta_final)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")



sample = np.array([[9,8,9.5,9,8,8,9.5,9,9.5,9.5,8,7,7,9,9,9,8,7,8,4,8,7,10,6,10,9,7,9.5,5]])

sample_scaled = scaler.transform(sample)
sample_with_intercept = np.c_[np.ones((sample_scaled.shape[0], 1)), sample_scaled]
probs = softmax(sample_with_intercept @ theta_final)

for label, prob in zip(class_map.keys(), probs[0]):
    print(f"{label}: {prob:.4f}")
