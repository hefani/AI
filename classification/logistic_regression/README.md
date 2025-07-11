# Logistic Regression using Batch Gradient Ascent
 
This project implements logistic regression from scratch using batch gradient ascent to classify raisin types (Besni vs Kecimen) based on two features: Eccentricity and MajorAxisLength.

---

# Overview

- The objective is to demonstrate understanding of binary classification using logistic regression
- is trained using numerical optimization (batch gradient ascent)

---

# DataSet

Raisin_Dataset.csv: Contains features like:
- Eccentricity
- MajorAxisLength
- ...and more (30 total)
- The target variable is Raisin Class (binary: "Besni" = 0, "Kecimen" = 1)

---

# Concepts Used

- Logistic function (sigmoid)
- Cost function (cross-entropy)
- Batch Gradient Ascent optimization
- Decision boundary visualization
- Model evaluation (accuracy)

---

# Files

- `logistic_regression.py` – Main implementation
- `Raisin_Dataset.csv` – Sample dataset
- `README.md` – Project documentation
- logistic_regression.png – Saved plot with decision boundary

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib
2. Run:
   ```bash
   python logistic_regression.py
