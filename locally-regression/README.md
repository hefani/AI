# locally-regression

This project implements **Locally Weighted Linear Regression**, a non-parametric machine learning algorithm. LWR fits a model around each query point using a weighted subset of the data, making it especially effective for datasets with non-linear patterns.

---

# Overview

- Uses a **Gaussian kernel** to assign weights to training points based on distance to the query.
- Fits a separate regression line for each query point using **weighted least squares**.
- Visualizes how local regression adapts to varying data trends.

---

# Concepts Used

- Instance-based learning  
- Local vs global modeling  
- Matrix operations with NumPy  
- Gaussian weighting  
- Data visualization with Matplotlib

---

# Files

- `locally_weighted_regression.py` – Core LWR implementation  
- `Salary_dataset.csv` – Sample dataset of salaries vs. years of experience  
- `locally_weighted_Regression.png` – Output visualization  
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas matplotlib
2. Run:
   python locally_weighted_regression.py
