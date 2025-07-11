# Gaussian Discriminant Analysis (GDA) – Personality Classification
 
This project implements Gaussian Discriminant Analysis (GDA) from scratch to classify synthetic personality types based on psychological traits. The classifier is tested on a data with features such as social energy, creativity, talkativeness, risk taking, and more.

---

# Overview

- It estimates the mean vector for each class, a shared covariance matrix, and prior probabilities
- GDA uses these parameters to compute the probability of a data point belonging to each class and assigns the label with the highest probability
- This repository contains a full implementation of the Gaussian Discriminant Analysis (GDA) algorithm for multi-class classification.
- It is applied to a synthetic personality traits dataset, where each class represents a personality type (e.g., introvert, extrovert, ambivert)
- The model computes class-wise means, a shared covariance matrix, and prior probabilities, then uses them to classify new inputs based on likelihoods

---

# DataSet

personality_synthetic_dataset.csv: Contains features like:
-Social energy
-Talkativeness
-Empathy
-Risk taking
-...and more (30 total)
The target variable is personality_type (e.g., introvert, extrovert, ambivert).
