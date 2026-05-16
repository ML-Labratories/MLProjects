
# Gold Price Prediction

A small regression project where the goal is to predict the daily price of the SPDR Gold ETF (GLD) using other major asset prices.

## Problem Statement

The idea is simple:  
given the daily closing prices of the S&P 500 (SPX), an oil ETF (USO), a silver ETF (SLV), and the EUR/USD exchange rate, try to predict the daily Gold ETF price (`GLD`).

## Dataset

- **Samples**: 2,290 trading days (2008â€“2018)  
- **Features**: 5 numeric crossâ€‘asset prices

| Feature | Description |
|---|---|
| Date | Trading date |
| SPX | S&P 500 index value |
| USO | United States Oil Fund (ETF) price |
| SLV | iShares Silver Trust ETF price |
| EUR/USD | Euro / US Dollar exchange rate |
| **GLD** | **SPDR Gold Trust ETF price (target, USD)** |

Some additional features were created during preprocessing.  
These include 5â€‘day rolling averages and 1â€‘day differences for SPX, USO, SLV and EUR/USD.  
The `Date` column was also used to extract **year, month, and day of week**.

## Project Structure

```
Gold Price Prediction/
â”œâ”€â”€ 01_eda.ipynb              # Exploratory data analysis
â”œâ”€â”€ 02_data_cleaning.ipynb    # Feature engineering and preprocessing
â”œâ”€â”€ 03_model_building.ipynb   # Training and evaluating regression models
â”œâ”€â”€ utils.py                  # Helper functions used across notebooks
â”œâ”€â”€ requirements.txt          # Project dependencies
â”œâ”€â”€ README.md                 # Project description
â””â”€â”€ data/
    â”œâ”€â”€ gold_price_data.csv
    â””â”€â”€ gold_price_cleaned.csv
```

## Results

Seven regression models were tested, plus a tuned Random Forest model.  
Because silver (SLV) and gold move very closely together, this turns out to be a relatively easy regression task and most models achieve **RÂ² above 0.9**.

| Model | MAE | RMSE | RÂ² | MAPE |
|---|---|---|---|---|
| **Random Forest (Tuned)** | **1.03** | **1.57** | **0.9955** | **0.0085** |
| Random Forest | 1.03 | 1.59 | 0.9954 | 0.0086 |
| Gradient Boosting | 1.48 | 2.11 | 0.9919 | 0.0122 |
| Decision Tree | 1.51 | 2.20 | 0.9912 | 0.0127 |
| KNN (k=5) | 3.79 | 5.78 | 0.9390 | 0.0314 |
| Linear Regression | 5.07 | 6.49 | 0.9232 | 0.0415 |
| Ridge | 5.05 | 6.49 | 0.9232 | 0.0413 |
| Lasso | 4.95 | 6.54 | 0.9221 | 0.0400 |

Best GridSearchCV parameters:  
`{max_depth: 20, min_samples_leaf: 1, n_estimators: 300}`  
(CV RÂ² â‰ˆ 0.993)

## Key Findings

- **Silver (SLV) is by far the strongest predictor of gold.**  
  The correlation between SLV and GLD is above 0.85. This makes sense because both are precious metals and often move together as â€œsafeâ€‘havenâ€ assets.

- **This project treats the dataset as a standard regression problem**, not a strict timeâ€‘series forecasting task. Each row is used as an independent `(X, y)` pair. Because sameâ€‘day SLV is included as a feature, the model indirectly gets strong information about the gold price.

- **The tuned Random Forest achieves a MAPE of about 0.85%**, which means the predictions are usually very close to the true value.

- **Linear models already perform fairly well (RÂ² â‰ˆ 0.92)**, which shows that the relationship between gold and silver is mostly linear.

- **Rolling averages and difference features provide only a small improvement.** With only ~2,300 rows, tree models can already capture most of the patterns from the raw features.

## Tech Stack

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikitâ€‘learn  

## Getting Started

Install dependencies and open the notebooks.
