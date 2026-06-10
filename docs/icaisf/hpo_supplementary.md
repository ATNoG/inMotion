# Supplementary Material: Hyperparameter Optimization Details

Best configurations obtained via Optuna with TPE sampler (50 trials per model, NSGA-II for SoftMoE).

## HPO GRU

| Parameter     | Value               |
| ------------- | ------------------- |
| hidden_size   | 552                 |
| num_layers    | 6                   |
| bidirectional | False               |
| attention     | False               |
| dropout       | 0.157               |
| loss_type     | focal               |
| focal_gamma   | 4.18                |
| lr            | 0.00119             |
| optimizer     | AdamW               |
| scheduler     | cosine              |
| weight_decay  | 9e-6                |
| L1            | 0                   |
| grad_clip     | 4.36                |
| Database      | `optuna_dl_9010.db` |
| Best val MCC  | 0.9030              |

## HPO TCN

| Parameter    | Value               |
| ------------ | ------------------- |
| num_channels | 363                 |
| depth        | 4                   |
| kernel_size  | 5                   |
| dropout      | 0.064               |
| loss_type    | CE                  |
| lr           | 0.000132            |
| optimizer    | AdamW               |
| scheduler    | cosine              |
| weight_decay | 2.7e-5              |
| L1           | 4.7e-5              |
| grad_clip    | 1.94                |
| Database     | `optuna_dl_5592.db` |
| Best val MCC | 0.8526              |

## HPO Mamba

| Parameter    | Value               |
| ------------ | ------------------- |
| d_model      | 256                 |
| num_layers   | 6                   |
| d_state      | 16                  |
| expand       | 4                   |
| mimo_rank    | 1                   |
| dropout      | 0.247               |
| loss_type    | CE                  |
| lr           | 0.000102            |
| optimizer    | AdamW               |
| scheduler    | cosine              |
| weight_decay | 0                   |
| L1           | 5e-5                |
| grad_clip    | 4.48                |
| Database     | `optuna_dl_9010.db` |
| Best val MCC | 0.8506              |

## HPO CNN

| Parameter    | Value               |
| ------------ | ------------------- |
| num_filters  | 218                 |
| num_blocks   | 12                  |
| kernel_set   | 2                   |
| dropout      | 0.185               |
| loss_type    | focal               |
| focal_gamma  | 2.96                |
| lr           | 0.00317             |
| optimizer    | AdamW               |
| scheduler    | cosine              |
| weight_decay | 3.4e-4              |
| L1           | 1e-6                |
| grad_clip    | 2.85                |
| Database     | `optuna_dl_5592.db` |
| Best val MCC | 0.8488              |

## HPO LSTM

| Parameter     | Value               |
| ------------- | ------------------- |
| hidden_size   | 382                 |
| num_layers    | 9                   |
| bidirectional | True                |
| attention     | True                |
| dropout       | 0.212               |
| loss_type     | focal               |
| focal_gamma   | 2.59                |
| lr            | 0.00124             |
| optimizer     | AdamW               |
| scheduler     | plateau             |
| weight_decay  | 0.00116             |
| L1            | 0                   |
| grad_clip     | 2.13                |
| Database      | `optuna_dl_5592.db` |
| Best val MCC  | 0.8388              |

## HPO BiLSTM

| Parameter    | Value               |
| ------------ | ------------------- |
| hidden_size  | 144                 |
| num_layers   | 5                   |
| attention    | True                |
| dropout      | 0.502               |
| loss_type    | CE                  |
| lr           | 0.00308             |
| optimizer    | AdamW               |
| scheduler    | plateau             |
| weight_decay | 1e-6                |
| L1           | 0                   |
| grad_clip    | 1.66                |
| Database     | `optuna_dl_5592.db` |
| Best val MCC | 0.8410              |

## HPO RNN

| Parameter     | Value          |
| ------------- | -------------- |
| hidden_size   | 305            |
| num_layers    | 3              |
| bidirectional | True           |
| dropout       | 0.153          |
| loss_type     | CE             |
| lr            | 0.00343        |
| optimizer     | SGD            |
| scheduler     | cosine         |
| momentum      | 0.903          |
| weight_decay  | 8.0e-5         |
| L1            | 6.2e-6         |
| grad_clip     | 2.998          |
| Database      | `optuna_dl.db` |
| Best val MCC  | 0.6402         |
