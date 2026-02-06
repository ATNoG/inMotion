# Hyperparameter Optimization Details

This directory provides supplementary material related to the hyperparameter optimization process used in the paper:

**"RSSI-Based Passenger Movement Classification for Non-Intrusive Public Transport Monitoring"**

The optimization was conducted using Optuna with a Tree-structured Parzen Estimator (TPE) sampler and 5-fold stratified cross-validation.

Contents:
- `optuna_summary.md`: number of trials and best cross-validation accuracy per classifier
- `optimal_hyperparameters.md`: best hyperparameter configurations found
- `search_spaces.md`: hyperparameter search ranges explored for each model family
