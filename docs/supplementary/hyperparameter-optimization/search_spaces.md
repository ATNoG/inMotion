# Hyperparameter Search Spaces

The following table summarizes the hyperparameter search ranges explored during
Optuna-based optimization.

| Classifier Family     | Parameter            | Range / Values                      | Scale         |
|-----------------------|----------------------|-------------------------------------|---------------|
| Tree Ensembles        | n_estimators         | [50, 300]                           | Linear        |
|                       | max_depth            | [3, 30]                             | Linear        |
|                       | min_samples_split    | [2, 20]                             | Linear        |
|                       | min_samples_leaf     | [1, 10]                             | Linear        |
|                       | max_features         | {sqrt, log2, None}                  | Categorical   |
| Boosting Methods      | learning_rate        | [0.01, 0.3]                         | Log           |
|                       | subsample            | [0.6, 1.0]                          | Linear        |
|                       | reg_alpha            | [1e-8, 10]                          | Log           |
|                       | reg_lambda           | [1e-8, 10]                          | Log           |
| SVC (RBF)             | C                    | [0.01, 100]                         | Log           |
|                       | gamma                | {scale, auto}                       | Categorical   |
| KNN                   | n_neighbors          | [1, 20]                             | Linear        |
|                       | weights              | {uniform, distance}                 | Categorical   |
|                       | metric               | {euclidean, manhattan, minkowski}   | Categorical   |
|                       | p                    | [1, 5]                              | Linear        |
| MLP                   | n_layers             | [1, 3]                              | Linear        |
|                       | n_units_per_layer    | [32, 256]                           | Linear        |
|                       | alpha                | [1e-5, 0.1]                         | Log           |
|                       | learning_rate_init   | [1e-4, 0.1]                         | Log           |
|                       | activation           | {relu, tanh}                        | Categorical   |
| Logistic Regression   | C                    | [0.001, 100]                        | Log           |
|                       | l1_ratio             | [0, 1]                              | Linear        |
