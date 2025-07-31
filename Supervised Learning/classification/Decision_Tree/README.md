# Decision Tree Classifier – Drug Recommendation
 
This project contains both a from-scratch implementation of a Decision Tree classifier using the Gini impurity metric and a comparison with scikit-learn’s DecisionTreeClassifier. The dataset used is the drug200.csv dataset from Kaggle. The model aims to predict the appropriate drug type based on patient features.

---

# Overview

- The objective is to demonstrate understanding of classification using a decision tree algorithm.
- Includes a complete from-scratch implementation of the CART algorithm with Gini impurity.
- Handles categorical feature encoding and basic preprocessing for medical dataset analysis.
- Tree is trained recursively by selecting the best split at each node to minimize impurity.
- Also includes a comparison with scikit-learn's built-in DecisionTreeClassifier.
- Outputs accuracy on the test set and a visual plot of the decision tree structure.

---

# DataSet

drug200.csv: Contains features like:
- Age: Patient age
- Sex: Male/Female
- BP: Blood pressure level (LOW, NORMAL, HIGH)
- Cholesterol: Cholesterol level (NORMAL, HIGH)
- Na_to_K: Sodium-to-potassium ratio in blood

The target variable is One of 5 drugs (drugA, drugB, drugC, drugX, drugY)

---

# Concepts Used

- Gini Impurity for evaluating split quality
- Recursive binary tree construction
- Feature thresholding for node splitting
- Majority class selection at leaf nodes
- Model evaluation using test accuracy
- Manual data preprocessing and encoding
- Comparison with scikit-learn's decision tree
- Tree visualization using plot_tree

---

# Files

- `Decision_Tree.py` – Main implementation
- `drug200.csv` – Sample dataset
- `README.md` – Project documentation
- `Decision_Tree.png` – Saved plot with decision boundary

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
2. Run:
   ```bash
   python Decision_Tree.py
