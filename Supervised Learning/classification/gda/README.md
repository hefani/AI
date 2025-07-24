# Gaussian Discriminant Analysis (GDA) Classifier for Personality Prediction
This repository contains an implementation of Gaussian Discriminant Analysis (GDA) from scratch in Python for multi-class classification. The model is applied on a synthetic personality dataset with features representing various personality traits to predict personality types.
 
---

# Overview

- This repository implements Gaussian Discriminant Analysis (GDA), a generative machine learning model for multi-class classification.
- GDA models the conditional distribution of features for each class as a multivariate Gaussian with a shared covariance matrix.
- The model estimates class-wise means, a shared covariance matrix, and class priors during training.
- Classification is done by evaluating posterior probabilities derived from the Gaussian likelihood and class priors.
- GDA is particularly effective when the class-conditional feature distributions approximately follow Gaussian distributions and shares covariance structure, making it suitable for problems like personality type prediction with continuous trait features.
  
---

# DataSet

personality_synthetic_dataset.csv: Contains features like:
- Social energy
- Talkativeness
- Empathy
- Risk taking
- ...and more (30 total)

The target variable is personality_type (e.g., introvert, extrovert, ambivert).

---

# Concepts Used

- Gaussian Discriminant Analysis (GDA): A generative classification model that assumes class-conditional data follows a multivariate normal distribution with shared covariance.
- Multivariate Normal Distribution: Used to model feature distribution per class with mean vectors and covariance matrix.
- Class Priors: Probabilities of each class estimated from training data frequencies.
- Maximum Likelihood Estimation (MLE): For estimating means, covariance matrix, and priors from data.
- Matrix Algebra: For efficient computation of inverse covariance and Mahalanobis distances.
- Feature Scaling using StandardScaler: Standardizes features to zero mean and unit variance to improve model stability.
- NumPy: Used for efficient numerical operations and vectorized computations.
- Probability Theory: Computing posterior probabilities based on Gaussian likelihoods and priors for classification.

---

# Files

- `GDA.py` – Main implementation
- `personality_synthetic_dataset.csv` – Sample dataset
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
4. Run:
   ```bash
   python GDA.py
