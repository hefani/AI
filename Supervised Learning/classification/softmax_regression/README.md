# Softmax Regression Projects
This repository showcases two applications of softmax (multinomial logistic) regression using gradient descent:
- Personality Type Classification: Predicts if a person is an Introvert, Extrovert, or Ambivert based on psychological features.
- Raisin Classification: Classifies raisin types (Kecimen or Besni) using geometrical features.
 
---

# Overview

- This repository showcases two machine learning models built using Softmax Regression 
- Softmax Regression is a supervised learning algorithm used for multiclass classification
- It extends logistic regression to handle multiple classes by predicting a probability distribution over all possible classes
- The algorithm learns separate weight vectors for each class and applies the softmax function to map the output to a probability simplex
- It is trained using cross-entropy loss and gradient descent, making it ideal for problems like sentiment analysis, personality classification where the output isn’t just binary
  
---

# DataSet

Raisin_Dataset.csv: Contains features like:
- Eccentricity
- MajorAxisLength
- ...and more

The target variable is Raisin Class (binary: "Besni" = 0, "Kecimen" = 1)

personality_synthetic_dataset.csv: Contains features like:
- Social energy
- Talkativeness
- Empathy
- Risk taking
- ...and more (30 total)

The target variable is personality_type (e.g., introvert, extrovert, ambivert).

---

# Concepts Used

- Multiclass Logistic Regression (Softmax)
- Gradient Descent Optimization
- One-hot Encoding
- Cost Function for Multiclass Classification
- Feature Scaling using StandardScaler
- NumPy for vectorized matrix operations

---

# Files

- `IntrovertExtrovertAmbivert.py` – Main implementation
- `softmaxRegression.py` – Main implementation
- `personality_synthetic_dataset.csv` – Sample dataset
- `Raisin_Dataset.csv` – Sample dataset
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
4. Run:
   ```bash
   python IntrovertExtrovertAmbivert.py
   python softmaxRegression.py
