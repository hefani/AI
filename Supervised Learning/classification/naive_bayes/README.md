# Naive Bayes Spam Classifier
 
This project implements a Naive Bayes classifier from scratch to detect spam emails based on their content.

---

# Overview

- It uses probabilistic modeling with Laplace smoothing to classify emails as spam or non-spam (ham)
- The vocabulary is enhanced using both the dataset and WordNet lexical database for better generalization
- Text preprocessing using tokenization and vocabulary filtering witch Converts raw email text into a list of normalized words
- Laplace smoothing to handle zero probabilities
  
---

# DataSet

spam_mail_classifier.csv : Contains emails
The target variable is Label of the email (binary: "ham" = 0, "spam" = 1)

---

# Concepts Used

- Natural Language Processing (NLP)
- Probabilistic Models
- Text Classification
- Laplace Smoothing
- Tokenization and Vocabulary Filtering
- Usage of external lexical databases (WordNet)

---

# Files

- `NaiveBayes_lapace_smoothing.py` – Main implementation
- `spam_mail_classifier.csv.csv` – Sample dataset
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas nltk
2. Download requirements:
   ```bash
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
4. Run:
   ```bash
   python logistic_regression.py
