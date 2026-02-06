# Optuna Hyperparameter Optimization Summary

The Tree-structured Parzen Estimator (TPE) sampler was employed with fixed random seeds to ensure reproducibility. The optimization objective was 5-fold stratified cross-validation accuracy on the training set.

| Classifier              | Trials | Best CV Accuracy |
|-------------------------|--------|------------------|
| Random Forest           | 1,303  | 89.08%           |
| Extra Trees             | 1,250  | 91.45%           |
| Gradient Boosting       | 1,170  | 87.51%           |
| XGBoost                 | 1,150  | 87.51%           |
| LightGBM                | 51     | 79.89%           |
| SVC (RBF)               | 950    | 92.22%           |
| KNN                     | 950    | 85.88%           |
| MLP                     | 950    | 90.68%           |
| Logistic Regression     | 900    | 85.97%           |
