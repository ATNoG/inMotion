# Optimal Hyperparameter Configurations

This document lists the best hyperparameter configurations identified through
Optuna optimization for each classifier.

---

## Random Forest

| Parameter              | Value |
|------------------------|-------|
| n_estimators           | 97    |
| max_depth              | 13    |
| min_samples_split      | 2     |
| min_samples_leaf       | 1     |
| max_features           | sqrt  |
| bootstrap              | False |

---

## Extra Trees

| Parameter              | Value |
|------------------------|-------|
| n_estimators           | 182   |
| max_depth              | 9     |
| min_samples_split      | 9     |
| min_samples_leaf       | 1     |
| max_features           | sqrt  |

---

## Gradient Boosting

| Parameter              | Value  |
|------------------------|--------|
| n_estimators           | 156    |
| learning_rate          | 0.0616 |
| max_depth              | 12     |
| min_samples_split      | 4      |
| min_samples_leaf       | 2      |
| subsample              | 0.950  |

---

## XGBoost

| Parameter              | Value        |
|------------------------|--------------|
| n_estimators           | 104          |
| learning_rate          | 0.253        |
| max_depth              | 13           |
| min_child_weight       | 2            |
| subsample              | 0.800        |
| colsample_bytree       | 0.714        |
| reg_alpha (L1)         | 2.66e-6      |
| reg_lambda (L2)        | 1.368        |

---

## LightGBM

| Parameter              | Value        |
|------------------------|--------------|
| n_estimators           | 83           |
| learning_rate          | 0.0333       |
| max_depth              | 8            |
| num_leaves             | 25           |
| min_child_samples      | 45           |
| subsample              | 0.970        |
| colsample_bytree       | 0.923        |
| reg_alpha (L1)         | 4.15e-5      |
| reg_lambda (L2)        | 1.02e-7      |

---

## SVC (RBF Kernel)

| Parameter | Value |
|----------|-------|
| C        | 0.634 |
| gamma    | scale |
| kernel   | RBF   |

---

## K-Nearest Neighbors

| Parameter     | Value      |
|--------------|------------|
| n_neighbors  | 4          |
| weights      | distance   |
| metric       | minkowski |
| p            | 5          |

---

## MLP Neural Network

| Parameter              | Value           |
|------------------------|-----------------|
| hidden_layer_sizes     | (249, 140, 95)  |
| activation             | tanh            |
| alpha (L2 penalty)     | 0.000991        |
| learning_rate          | constant        |
| learning_rate_init     | 0.00696         |
| max_iter               | 500             |
| early_stopping         | True            |

---

## Logistic Regression

| Parameter      | Value |
|---------------|-------|
| C             | 36.81 |
| l1_ratio      | 0.214 |
| solver        | saga  |
| max_iter      | 1000  |
