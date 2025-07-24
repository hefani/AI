# Logistic Regression Models on Raisin Dataset
This repository contains two implementations of Logistic Regression algorithms for binary classification applied to the Raisin Dataset. Both models classify between two types of raisins ("Besni" and "Kecimen") based on their physical features.

# Overview

- Two logistic regression models are implemented from scratch using NumPy
- Batch Gradient Ascent: Iteratively updates parameters to maximize likelihood.
- Newton's Method: Uses second-order derivative (Hessian) information for faster convergence.
- Both models use the sigmoid (logistic) function to map linear model outputs to probabilities.
- The dataset contains physical measurements of raisins such as Area, Major Axis Length, Minor Axis Length, and others.
- Models include an intercept term and optimize parameters to separate the two classes.
- Visualization of decision boundary is provided for the 2D feature case (Gradient Ascent model).
  
---

# DataSet

Raisin_Dataset.csv: Contains features like:
- Eccentricity
- MajorAxisLength
- ...and more

---

# Concepts Used

- Logistic (sigmoid) function for probability estimation.
- Binary cross-entropy loss (log loss) as the cost function.
- Batch Gradient Ascent optimization.
- Newton’s method leveraging Hessian matrix for parameter updates.
- Feature engineering with intercept term.
- Model evaluation using accuracy.
- Visualization of decision boundary in 2D feature space (for first model).

---

# Files

- `Newton'sMethod.py` – Main implementation
- `logisticRegression.py` – Main implementation
- `Raisin_Dataset.csv` – Sample dataset
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
4. Run:
   ```bash
   python Newton'sMethod.py
   python logisticRegression.py
