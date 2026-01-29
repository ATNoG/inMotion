# WiFi Fingerprinting Classification Analysis Report

## Overview

This report documents the comprehensive machine learning classification analysis performed on the WiFi fingerprinting dataset from the inMotion project. The goal is to classify passenger movements (routes) based on RSSI (Received Signal Strength Indicator) patterns from WiFi signals.

## Dataset Description

### Data Collection Context

The dataset was collected in a controlled environment simulating a bus and bus stop scenario:
- **Zone A**: A room simulating the interior of a bus
- **Zone B**: A corridor simulating a bus stop

### Features

The dataset contains RSSI measurements over 10 time steps (1 second intervals):
- **Columns 1-10**: RSSI values (dBm) captured at each second
- **mac**: Device MAC address identifier
- **label**: Route class (AA, AB, BA, BB)
- **noise_label**: Boolean indicating if data was collected with interference

### Target Classes

| Class | Description | Meaning |
|-------|-------------|---------|
| AA | A → A | Staying inside the bus |
| AB | A → B | Exiting the bus (disembarking) |
| BA | B → A | Entering the bus (boarding) |
| BB | B → B | Staying at the bus stop |

## Project Structure

```
ml_classification/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings
├── data_loader.py        # Data loading and preprocessing
├── eda.py                # Exploratory Data Analysis
├── classifiers.py        # Classifier factory with all algorithms
├── optimization.py       # Optuna hyperparameter optimization
├── training.py           # Training pipeline
├── visualization.py      # Visualization utilities
└── utils.py              # Utility functions

run_classification.py     # Main classification script
run_optuna_optimization.py # Optuna optimization with dashboard
```

## Classifiers Implemented

### Tree-Based Classifiers
- Decision Tree Classifier
- Extra Tree Classifier

### Ensemble Methods
- Random Forest Classifier
- Extra Trees Classifier
- Bagging Classifier
- AdaBoost Classifier
- Voting Ensemble (soft voting)
- Stacking Ensemble

### Gradient Boosting Methods
- Gradient Boosting Classifier
- Histogram-based Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### Linear Classifiers (with Regularization)
- Logistic Regression (L1, L2, ElasticNet)
- Ridge Classifier
- SGD Classifier
- Perceptron
- Passive Aggressive Classifier

### Support Vector Machines
- SVC with RBF kernel
- SVC with Linear kernel
- SVC with Polynomial kernel
- Linear SVC
- Nu-SVC

### Neighbors-Based Classifiers
- K-Nearest Neighbors (k=3, 5, 7)
- Nearest Centroid

### Neural Networks
- MLP Small (50 neurons)
- MLP Medium (100, 50 neurons)
- MLP Large (128, 64, 32 neurons)
- MLP with L2 regularization

### Discriminant Analysis
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

### Other
- Gaussian Naive Bayes
- Gaussian Process Classifier
- Dummy Classifiers (baseline)

## Methodology

### Data Preprocessing
1. Load CSV data with pandas
2. Extract RSSI features (columns 1-10)
3. Encode labels using LabelEncoder
4. StandardScaler normalization
5. Stratified train/test split (80/20)

### Evaluation Strategy
- **Cross-Validation**: Stratified K-Fold (5 folds)
- **Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score
- **Reproducibility**: Random seed = 42

### Regularization Techniques
- L1 regularization (Lasso) for sparse feature selection
- L2 regularization (Ridge) for weight decay
- ElasticNet (combined L1/L2)
- Dropout and early stopping for neural networks
- Tree pruning parameters for ensemble methods

## Hyperparameter Optimization

Optuna is used for Bayesian hyperparameter optimization with the following features:
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 50 per classifier
- **Objective**: Maximize cross-validation accuracy
- **Storage**: SQLite database for dashboard visualization

### Search Spaces

Each classifier has a tailored search space covering:
- Model complexity (depth, estimators, neurons)
- Learning rates
- Regularization parameters
- Kernel parameters (for SVM)
- Distance metrics (for KNN)

## How to Run

### Basic Classification (All Classifiers)

```bash
uv run python run_classification.py
```

### Quick Mode (Subset of Classifiers)

```bash
uv run python run_classification.py --quick
```

### With Hyperparameter Optimization

```bash
uv run python run_classification.py --optimize --n-trials 50
```

### Dedicated Optuna Optimization with Dashboard

```bash
uv run python run_optuna_optimization.py --dashboard
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to dataset | dataset.csv |
| `--quick` | Use fewer classifiers | False |
| `--optimize` | Run Optuna optimization | False |
| `--n-trials` | Optuna trials per classifier | 50 |
| `--cv-folds` | Cross-validation folds | 5 |
| `--seed` | Random seed | 42 |
| `--n-jobs` | Parallel jobs (-1 = all) | -1 |
| `--skip-eda` | Skip exploratory analysis | False |

## Output Files

### Results Directory (`results/`)
- `classification_results.csv`: All classifier performance metrics
- `classification_report.txt`: Detailed text report
- `optuna_optimization_results.txt`: Hyperparameter tuning results

### Models Directory (`models/`)
- `{classifier_name}.joblib`: Saved trained models

### Plots Directory (`plots/`)
- `eda/`: Exploratory data analysis plots
  - `class_distribution.png`
  - `rssi_distributions.png`
  - `correlation_heatmap.png`
  - `pca_visualization.png`
  - `tsne_visualization.png`
  - `class_trajectories.png`
- `results/`: Classification results plots
  - `classifier_comparison_accuracy.png`
  - `multi_metric_comparison.png`
  - `cv_scores.png`
  - `training_times.png`
  - `accuracy_vs_time.png`
  - `metric_heatmap.png`
  - `confusion_matrix_{name}.png`
  - `feature_importance.png`
  - `mean_feature_importance.png`

## Exploratory Data Analysis Insights

### Dataset Statistics
- Total samples varies based on collection
- 10 RSSI features (temporal sequence)
- 4 target classes representing movement patterns
- Multiple device MAC addresses

### Key Observations
1. **Temporal patterns**: RSSI values show distinct trajectories for different routes
2. **Class separability**: PCA shows reasonable cluster separation
3. **Feature correlations**: Adjacent time steps are highly correlated
4. **Signal characteristics**: 
   - AA/BB (stationary): More stable RSSI patterns
   - AB/BA (transitional): Changing RSSI as device moves

## Feature Importance Analysis

The temporal sequence of RSSI values carries different predictive power:
- **Early time steps (1-3)**: Capture initial position
- **Middle time steps (4-7)**: Capture transition
- **Late time steps (8-10)**: Capture final position

For transitional classes (AB, BA), the gradient across time steps is most informative.

## Model Persistence

All trained models are saved using `joblib` for:
- Future predictions
- Model deployment
- Ensemble creation
- Further analysis

Load a model:
```python
import joblib
model = joblib.load("models/RandomForest.joblib")
predictions = model.predict(X_new)
```

## Dependencies

Required packages (managed via `uv`):
- scikit-learn
- xgboost
- lightgbm
- catboost
- optuna
- optuna-dashboard
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- imbalanced-learn

## Reproducibility

- Random seeds set for all stochastic operations
- Cross-validation ensures robust evaluation
- All parameters logged in results files
- Git version control for code tracking

## Future Improvements

1. **Feature Engineering**: 
   - Add gradient features (RSSI change rate)
   - Statistical aggregations (mean, std, range)
   - Frequency domain features (FFT)

2. **Advanced Models**:
   - Recurrent Neural Networks (LSTM, GRU)
   - 1D Convolutional Networks
   - Attention mechanisms

3. **Data Augmentation**:
   - Time shifting
   - Noise injection
   - Interpolation

4. **Multi-device Fusion**:
   - Combine signals from multiple devices
   - Graph-based learning

## Conclusion

This comprehensive ML classification analysis provides:
- Evaluation of 30+ classifiers
- Automated hyperparameter optimization
- Detailed performance metrics and visualizations
- Saved models ready for deployment
- Complete documentation for reproducibility

The results can be used to select the best model for the WiFi fingerprinting passenger tracking system in the inMotion project.
