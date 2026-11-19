# Predicting Loan Default: Machine Learning vs a Neural Network

Comparison of Decision Tree, Random Forest, Gaussian Naive Bayes, Logistic Regression and a
multilayer neural network (Keras) for predicting whether a LendingClub loan will be repaid.

## Problem
Lenders lose far more from a missed defaulter than from a declined good borrower. Only 16% of the
loans in this data were not fully repaid, so a model that always predicts "repaid" is 84% accurate
and catches no defaulters. Accuracy alone cannot choose the best model.

## Data
[Kaggle: Loan data (Suru)](https://www.kaggle.com/datasets/itssuru/loan-data), 9,578 loans issued
2007-2010, 13 features, target `not.fully.paid`. The CSV is not included: download `loan_data.csv`
and place it next to the notebook (check the dataset's licence on Kaggle).

## Approach
- One-hot encoded loan purpose; engineered instalment-to-income ratio and credit history in years.
- Stratified 80/20 split (7,662 train / 1,916 test), `StandardScaler` fitted on training data only.
- SMOTE applied to the training set only, so no test information leaks into training.
- Four classical models tuned with grid search and 5-fold stratified cross-validation (scored on F1).
- Neural network: 3 hidden layers (64-32-16), ReLU, dropout 0.3, Adam, early stopping.
- Metrics: accuracy, precision, recall, F1 and ROC-AUC on the held-out test set.

## Results (test set: 1,916 loans, 307 defaulters)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.635 | 0.239 | 0.586 | 0.340 | 0.681 |
| Naive Bayes | 0.654 | 0.244 | 0.550 | 0.338 | 0.663 |
| Neural network (MLP) | 0.637 | 0.216 | 0.479 | 0.297 | 0.631 |
| Random Forest | 0.773 | 0.287 | 0.280 | 0.283 | 0.661 |
| Decision Tree | 0.780 | 0.266 | 0.212 | 0.236 | 0.582 |

## Key findings
1. **Accuracy misleads.** The untuned Random Forest scored about 84% accuracy but caught 3 of 307 defaulters.
2. **SMOTE made the models useful.** Random Forest recall rose from 1% to 28%; Logistic Regression from under 5% to about 59%, at the cost of accuracy.
3. **Simpler models won.** Logistic Regression and Naive Bayes had the best F1 (about 0.34); the neural network did not beat them, consistent with research on small tabular data (Grinsztajn et al., 2022).
4. **Signal is moderate.** ROC-AUC ranged 0.58-0.68. Recent credit inquiries, FICO score and interest rate were the strongest predictors.

## Limitations
Small dataset (2007-2010 loans), moderate predictive signal, no cost-sensitive threshold tuning, and results
depend on SMOTE's synthetic records. Future work: threshold tuning by cost, gradient boosting, richer borrower data.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook loan_default_ml_vs_ann.ipynb
```

## About this project
University group project (3 members: Kalyan Acharya, Nabin Thapa, Ritesh Panta).
My contribution: [describe your parts, e.g. preprocessing, SMOTE and the classical ML models].
[Note any AI assistance honestly.]
