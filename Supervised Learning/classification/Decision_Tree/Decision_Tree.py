import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None
        current_gini = self._gini(y)
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)
                gain = current_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = t
        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_classes = len(y), len(np.unique(y))

        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples):
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=self._majority_class(y))

        left_idxs = X[:, feature] <= threshold
        right_idxs = X[:, feature] > threshold
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature, threshold, left, right)

    def _majority_class(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv("drug200.csv")

sex_map = {"F": 0, "M": 1}
bp_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
chol_map = {"NORMAL": 0, "HIGH": 1}

df["Sex"] = df["Sex"].map(sex_map)
df["BP"] = df["BP"].map(bp_map)
df["Cholesterol"] = df["Cholesterol"].map(chol_map)

features = ["Age", "Sex" , "BP", "Cholesterol", "Na_to_K"]
X = df[features].values
class_map = {label: idx for idx, label in enumerate(df["Drug"].unique())}
y = df["Drug"].map(class_map).values

X, y = X[y != 2], y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


model = DecisionTreeClassifier()
model.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=features, class_names=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
plt.title("Decision Tree")
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.grid()
plt.savefig("Decision_Tree.png")
