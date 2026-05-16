# Red Wine Quality Classification

A machine learning project that predicts whether a red wine is **high quality (quality ≥ 7)** using physicochemical measurements from the UCI Wine Quality dataset.

The project explores the relationship between wine chemistry and sensory ratings while building classification models to identify high‑quality wines.

---

## Problem

Wine quality is usually determined by professional tasters, but chemical measurements of wine can reveal patterns related to quality.

In this project, the original quality score (0–10) is converted into a binary classification task:

- **1 → Good quality wine (quality ≥ 7)**
- **0 → Not good wine (quality < 7)**

Because only about **13.6% of the wines are labeled as good**, the dataset is imbalanced.  
For this reason, metrics such as **F1 score, precision, and recall** are more informative than accuracy alone.

---

## Dataset

Source: **UCI Machine Learning Repository – Wine Quality Dataset**  
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

Dataset characteristics:

- **1,599 red wine samples**
- **11 physicochemical features**
- **1 target variable (wine quality score)**

Main input features include acidity levels, sulfur dioxide measurements, density, pH, sulphates, and alcohol content.

---

## Project Workflow

The project follows a typical machine learning pipeline:

1. **Exploratory Data Analysis (EDA)**  
   Understanding distributions, correlations, and feature relationships.

2. **Data Cleaning & Feature Engineering**  
   - Creating the binary target (`good_quality`)
   - Engineering chemistry‑inspired features
   - Encoding categorical bins

3. **Model Training & Evaluation**  
   Training several classification models and comparing their performance.

4. **Hyperparameter Tuning**  
   Improving the best model using **GridSearchCV**.

---

## Models Tested

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K‑Nearest Neighbors  
- Support Vector Machine (SVM)  
- Gradient Boosting  
- Naive Bayes  

Tree‑based ensemble methods performed best overall.

---

## Key Insights

Several chemical properties strongly influence wine quality:

- **Alcohol** is the strongest positive predictor of good wine.
- **Volatile acidity** negatively affects wine quality.
- **Sulphates** and **citric acid** tend to increase quality ratings.
- **Density** and high sulfur levels often correlate with lower ratings.

Random Forest achieved the most reliable performance due to its ability to model non‑linear relationships between features.

---

## Project Structure

```
Wine Quality Classification
│
├── 01_eda.ipynb
├── 02_data_cleaning.ipynb
├── 03_model_building.ipynb
├── utils.py
├── requirements.txt
├── README.md
└── data
    ├── winequality-red.csv
    └── winequality-red_cleaned.csv
```

Run the notebooks in order:

```
01_eda.ipynb → 02_data_cleaning.ipynb → 03_model_building.ipynb
```

---

## Tech Stack

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit‑learn  

---

## Future Improvements

Possible next steps include:

- Applying **SMOTE** to address class imbalance
- Training models on the **white wine dataset**
- Using **SHAP values** for model explainability
- Building a **Streamlit app** for interactive predictions

---
