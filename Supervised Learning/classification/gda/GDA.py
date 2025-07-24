import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class GDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means = {}
        self.priors = {}
        self.cov = np.zeros((n_features, n_features))
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]
            self.cov += np.cov(X_cls, rowvar=False) * (X_cls.shape[0] - 1)
        
        self.cov /= (X.shape[0] - len(self.classes))
        self.inv_cov = np.linalg.inv(self.cov)
    
    def predict(self, X):
        scores = []
        for cls in self.classes:
            mean = self.means[cls]
            prior = self.priors[cls]
            score = X @ self.inv_cov @ mean - 0.5 * mean.T @ self.inv_cov @ mean + np.log(prior)
            scores.append(score)
        return self.classes[np.argmax(scores, axis=0)]

    def predict_proba(self, X):
        probs = []
        for cls in self.classes:
            mean = self.means[cls]
            prior = self.priors[cls]
            exponent = -0.5 * np.sum((X - mean) @ self.inv_cov * (X - mean), axis=1)
            likelihood = np.exp(exponent)
            probs.append(prior * likelihood)
        probs = np.array(probs).T
        return probs / probs.sum(axis=1, keepdims=True)


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

model = GDA()
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
print("Accuracy:", accuracy_score(y, y_pred))



sample = np.array([[9,8,9.5,9,8,8,9.5,9,9.5,9.5,8,7,7,9,9,9,8,7,8,4,8,7,10,6,10,9,7,9.5,5]])
sample_scaled = scaler.transform(sample)
probs = model.predict_proba(sample_scaled)
predicted_class = np.argmax(probs, axis=1)[0] 
for label, prob in zip(class_map.keys(), probs[0]):
    print(f"{label}: {prob:.4f}")

