
## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.6391 | 0.7273 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.5976 | 0.6962 |
| HPO_BILSTM | hpo | cuda:0 | 0.5879 | 0.6896 |
| DS_BiLSTM_A | ds_base | cuda:0 | 0.5790 | 0.6829 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.5762 | 0.6807 |
| HPO_TRANSFORMER | hpo | cuda:1 | 0.5697 | 0.6763 |
| DS_Mamba_A | ds_base | cuda:0 | 0.5671 | 0.6718 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.5671 | 0.6741 |
| DS_LSTM_A | ds_base | cuda:0 | 0.5665 | 0.6741 |
| DS_Mamba_B | ds_base | cuda:1 | 0.5646 | 0.6718 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.5644 | 0.6718 |
| Ensemble_BiLSTM_LSTM | voting_ensemble | cuda:0 | 0.5640 | 0.6718 |
| MoE_4GRU | soft_moe | cuda:0 | 0.5639 | 0.6718 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.5634 | 0.6718 |
| BiLSTM | single | cuda:0 | 0.5622 | 0.6696 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.5618 | 0.6696 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.5611 | 0.6696 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.5609 | 0.6696 |
| HPO_MAMBA | hpo | cuda:0 | 0.5605 | 0.6696 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.5603 | 0.6674 |
| MoE_4CNN | soft_moe | cuda:1 | 0.5588 | 0.6674 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.5578 | 0.6674 |
| LSTM | single | cuda:0 | 0.5547 | 0.6652 |
| TCN | single | cuda:0 | 0.5545 | 0.6630 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.5539 | 0.6630 |
| DS_LSTM_B | ds_base | cuda:1 | 0.5512 | 0.6630 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.5512 | 0.6585 |
| GRU | single | cuda:1 | 0.5506 | 0.6608 |
| Ensemble_TCN_Transformer | voting_ensemble | cuda:0 | 0.5494 | 0.6608 |
| DS_GRU_A | ds_base | cuda:0 | 0.5488 | 0.6608 |
| DS_BiLSTM_B | ds_base | cuda:1 | 0.5483 | 0.6608 |
| DS_RNN_A | ds_base | cuda:0 | 0.5481 | 0.6585 |
| DS_TCN_B | ds_base | cuda:1 | 0.5457 | 0.6585 |
| HPO_RNN | hpo | cuda:0 | 0.5450 | 0.6585 |
| RNN | single | cuda:0 | 0.5444 | 0.6563 |
| DS_TCN_A | ds_base | cuda:0 | 0.5436 | 0.6563 |
| MoE_Mixed | soft_moe | cuda:0 | 0.5435 | 0.6563 |
| CNN | single | cuda:1 | 0.5433 | 0.6563 |
| Transformer | single | cuda:1 | 0.5429 | 0.6563 |
| HPO_LSTM | hpo | cuda:0 | 0.5421 | 0.6519 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.5407 | 0.6541 |
| DS_CNN_B | ds_base | cuda:1 | 0.5401 | 0.6541 |
| CNN2DLSTM | single | cuda:1 | 0.5396 | 0.6541 |
| HPO_GRU | hpo | cuda:1 | 0.5385 | 0.6475 |
| HPO_CNN | hpo | cuda:1 | 0.5385 | 0.6519 |
| CNN2DGRU | single | cuda:0 | 0.5360 | 0.6519 |
| DS_Transformer_B | ds_base | cuda:1 | 0.5358 | 0.6497 |
| DS_CNN2DRNN_A | ds_base | cuda:0 | 0.5352 | 0.6497 |
| DS_CNN_A | ds_base | cuda:0 | 0.5335 | 0.6475 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.5321 | 0.6452 |
| DS_RNN_B | ds_base | cuda:1 | 0.5311 | 0.6452 |
| DS_Transformer_A | ds_base | cuda:0 | 0.5298 | 0.6408 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.5295 | 0.6430 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.5261 | 0.6430 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.5250 | 0.6430 |
| Mamba | single | cuda:1 | 0.5246 | 0.6408 |
| MoE_4TCN | soft_moe | cuda:0 | 0.5230 | 0.6408 |
| DS_GRU_B | ds_base | cuda:1 | 0.5190 | 0.6364 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.5179 | 0.6364 |
| HPO_TCN | hpo | cuda:0 | 0.5161 | 0.6364 |
| HPO_CNN2DRNN | hpo | cuda:1 | 0.5011 | 0.6253 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-simple-2`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| DeepStackEnsemble | deep_stack | cuda:0 | 0.8585 | 0.8937 |
| HPO_GRU | hpo | cuda:1 | 0.8524 | 0.8893 |
| MoE_4TCN | soft_moe | cuda:0 | 0.8521 | 0.8883 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.8499 | 0.8873 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.8488 | 0.8858 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.8472 | 0.8853 |
| HPO_TCN | hpo | cuda:0 | 0.8470 | 0.8843 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.8459 | 0.8838 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.8455 | 0.8838 |
| DS_CNN_B | ds_base | cuda:1 | 0.8447 | 0.8828 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.8446 | 0.8824 |
| DS_TCN_A | ds_base | cuda:0 | 0.8442 | 0.8824 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.8436 | 0.8819 |
| HPO_LSTM | hpo | cuda:0 | 0.8435 | 0.8824 |
| DS_CNN_A | ds_base | cuda:0 | 0.8423 | 0.8809 |
| HPO_MAMBA | hpo | cuda:0 | 0.8418 | 0.8809 |
| MoE_4CNN | soft_moe | cuda:1 | 0.8405 | 0.8799 |
| CNN | single | cuda:1 | 0.8383 | 0.8774 |
| HPO_CNN | hpo | cuda:1 | 0.8357 | 0.8754 |
| TCN | single | cuda:0 | 0.8356 | 0.8759 |
| MoE_Mixed | soft_moe | cuda:0 | 0.8353 | 0.8759 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.8330 | 0.8744 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.8329 | 0.8744 |
| DS_RNN_B | ds_base | cuda:1 | 0.8322 | 0.8735 |
| HPO_RNN | hpo | cuda:0 | 0.8319 | 0.8739 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.8265 | 0.8685 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.8252 | 0.8685 |
| HPO_BILSTM | hpo | cuda:0 | 0.8195 | 0.8641 |
| HPO_CNN2DRNN | hpo | cuda:1 | 0.8188 | 0.8631 |
| DS_Mamba_B | ds_base | cuda:1 | 0.8185 | 0.8636 |
| MoE_4GRU | soft_moe | cuda:0 | 0.8173 | 0.8626 |
| DS_CNN2DRNN_A | ds_base | cuda:0 | 0.8140 | 0.8601 |
| DS_Transformer_A | ds_base | cuda:0 | 0.8094 | 0.8566 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.8074 | 0.8552 |
| DS_TCN_B | ds_base | cuda:1 | 0.8051 | 0.8532 |
| HPO_TRANSFORMER | hpo | cuda:1 | 0.8035 | 0.8522 |
| DS_Mamba_A | ds_base | cuda:0 | 0.8019 | 0.8512 |
| CNN2DLSTM | single | cuda:1 | 0.8010 | 0.8502 |
| Mamba | single | cuda:1 | 0.8002 | 0.8497 |
| Ensemble_TCN_Transformer | voting_ensemble | cuda:0 | 0.7979 | 0.8482 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7926 | 0.8443 |
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.7772 | 0.8324 |
| CNN2DGRU | single | cuda:0 | 0.7754 | 0.8314 |
| DS_RNN_A | ds_base | cuda:0 | 0.7634 | 0.8220 |
| RNN | single | cuda:0 | 0.7552 | 0.8161 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.7451 | 0.8082 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.7434 | 0.8067 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.7361 | 0.8013 |
| DS_Transformer_B | ds_base | cuda:1 | 0.7258 | 0.7939 |
| DS_GRU_A | ds_base | cuda:0 | 0.7247 | 0.7929 |
| Transformer | single | cuda:1 | 0.7134 | 0.7850 |
| DS_GRU_B | ds_base | cuda:1 | 0.7129 | 0.7840 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7104 | 0.7820 |
| GRU | single | cuda:1 | 0.7101 | 0.7815 |
| DS_LSTM_A | ds_base | cuda:0 | 0.6989 | 0.7736 |
| DS_BiLSTM_A | ds_base | cuda:0 | 0.6937 | 0.7696 |
| Ensemble_BiLSTM_LSTM | voting_ensemble | cuda:0 | 0.6922 | 0.7687 |
| DS_BiLSTM_B | ds_base | cuda:1 | 0.6899 | 0.7657 |
| LSTM | single | cuda:0 | 0.6855 | 0.7637 |
| BiLSTM | single | cuda:0 | 0.6838 | 0.7622 |
| DS_LSTM_B | ds_base | cuda:1 | 0.6820 | 0.7603 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-augmented-2`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.6391 | 0.7273 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.5976 | 0.6962 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.5812 | 0.6829 |
| HPO_MAMBA | hpo | cuda:0 | 0.5716 | 0.6785 |
| DS_GRU_A | ds_base | cuda:0 | 0.5706 | 0.6763 |
| HPO_BILSTM | hpo | cuda:0 | 0.5693 | 0.6763 |
| HPO_TRANSFORMER | hpo | cuda:1 | 0.5660 | 0.6718 |
| HPO_GRU | hpo | cuda:1 | 0.5649 | 0.6718 |
| MoE_4GRU | soft_moe | cuda:0 | 0.5625 | 0.6696 |
| DS_BiLSTM_A | ds_base | cuda:0 | 0.5615 | 0.6696 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.5614 | 0.6674 |
| DS_Transformer_B | ds_base | cuda:1 | 0.5612 | 0.6696 |
| Ensemble_TCN_Transformer | voting_ensemble | cuda:0 | 0.5611 | 0.6696 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.5607 | 0.6696 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.5602 | 0.6696 |
| Ensemble_BiLSTM_LSTM | voting_ensemble | cuda:0 | 0.5598 | 0.6696 |
| DS_LSTM_A | ds_base | cuda:0 | 0.5577 | 0.6674 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.5574 | 0.6674 |
| BiLSTM | single | cuda:0 | 0.5566 | 0.6674 |
| LSTM | single | cuda:0 | 0.5548 | 0.6652 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.5548 | 0.6652 |
| HPO_CNN | hpo | cuda:1 | 0.5546 | 0.6652 |
| MoE_4TCN | soft_moe | cuda:0 | 0.5545 | 0.6652 |
| DS_LSTM_B | ds_base | cuda:1 | 0.5541 | 0.6652 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.5531 | 0.6630 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.5530 | 0.6630 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.5522 | 0.6630 |
| GRU | single | cuda:1 | 0.5519 | 0.6630 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.5518 | 0.6630 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.5491 | 0.6608 |
| DS_Mamba_A | ds_base | cuda:0 | 0.5484 | 0.6608 |
| HPO_CNN2DRNN | hpo | cuda:1 | 0.5477 | 0.6585 |
| MoE_4CNN | soft_moe | cuda:1 | 0.5473 | 0.6585 |
| TCN | single | cuda:0 | 0.5469 | 0.6585 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.5455 | 0.6585 |
| CNN2DGRU | single | cuda:0 | 0.5455 | 0.6585 |
| DS_CNN2DRNN_A | ds_base | cuda:0 | 0.5452 | 0.6541 |
| HPO_TCN | hpo | cuda:0 | 0.5436 | 0.6563 |
| DS_BiLSTM_B | ds_base | cuda:1 | 0.5431 | 0.6541 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.5425 | 0.6563 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.5423 | 0.6563 |
| DS_TCN_A | ds_base | cuda:0 | 0.5416 | 0.6563 |
| CNN2DLSTM | single | cuda:1 | 0.5403 | 0.6541 |
| HPO_LSTM | hpo | cuda:0 | 0.5399 | 0.6541 |
| DS_TCN_B | ds_base | cuda:1 | 0.5396 | 0.6541 |
| DS_RNN_B | ds_base | cuda:1 | 0.5392 | 0.6541 |
| CNN | single | cuda:1 | 0.5391 | 0.6541 |
| RNN | single | cuda:0 | 0.5390 | 0.6519 |
| DS_RNN_A | ds_base | cuda:0 | 0.5359 | 0.6497 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.5349 | 0.6497 |
| Mamba | single | cuda:1 | 0.5346 | 0.6497 |
| DS_CNN_B | ds_base | cuda:1 | 0.5343 | 0.6497 |
| DS_CNN_A | ds_base | cuda:0 | 0.5327 | 0.6475 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.5320 | 0.6452 |
| HPO_RNN | hpo | cuda:0 | 0.5276 | 0.6452 |
| DS_Mamba_B | ds_base | cuda:1 | 0.5259 | 0.6408 |
| Transformer | single | cuda:1 | 0.5220 | 0.6408 |
| DS_GRU_B | ds_base | cuda:1 | 0.5215 | 0.6386 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.5208 | 0.6341 |
| DS_Transformer_A | ds_base | cuda:0 | 0.5182 | 0.6386 |
| MoE_Mixed | soft_moe | cuda:0 | 0.5031 | 0.6253 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-normal`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| DS_CNN_A | ds_base | cuda:0 | 0.7677 | 0.8250 |
| DS_RNN_A | ds_base | cuda:0 | 0.7674 | 0.8250 |
| Mamba | single | cuda:1 | 0.7583 | 0.8167 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.7550 | 0.8125 |
| MoE_4TCN | soft_moe | cuda:0 | 0.7550 | 0.8125 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.7533 | 0.8125 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.7527 | 0.8125 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.7526 | 0.8125 |
| MoE_4GRU | soft_moe | cuda:0 | 0.7522 | 0.8125 |
| HPO_CNN | hpo | cuda:1 | 0.7511 | 0.8125 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.7489 | 0.8083 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.7487 | 0.8083 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7445 | 0.8042 |
| HPO_LSTM | hpo | cuda:0 | 0.7425 | 0.8042 |
| DS_CNN_B | ds_base | cuda:1 | 0.7421 | 0.8042 |
| CNN | single | cuda:1 | 0.7413 | 0.8042 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.7407 | 0.8042 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.7378 | 0.8000 |
| DS_Mamba_A | ds_base | cuda:0 | 0.7362 | 0.8000 |
| TCN | single | cuda:0 | 0.7347 | 0.7958 |
| HPO_TCN | hpo | cuda:0 | 0.7324 | 0.7958 |
| GRU | single | cuda:1 | 0.7323 | 0.7958 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.7306 | 0.7958 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.7297 | 0.7958 |
| DS_TCN_A | ds_base | cuda:0 | 0.7295 | 0.7917 |
| RNN | single | cuda:0 | 0.7290 | 0.7958 |
| DS_LSTM_B | ds_base | cuda:1 | 0.7288 | 0.7958 |
| HPO_GRU | hpo | cuda:1 | 0.7286 | 0.7958 |
| DS_GRU_A | ds_base | cuda:0 | 0.7285 | 0.7958 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7273 | 0.7917 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.7266 | 0.7917 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.7260 | 0.7917 |
| DS_CNN2DRNN_A | ds_base | cuda:0 | 0.7240 | 0.7875 |
| HPO_TRANSFORMER | hpo | cuda:1 | 0.7237 | 0.7917 |
| DS_BiLSTM_B | ds_base | cuda:1 | 0.7216 | 0.7875 |
| CNN2DLSTM | single | cuda:1 | 0.7214 | 0.7875 |
| DS_TCN_B | ds_base | cuda:1 | 0.7213 | 0.7875 |
| DS_GRU_B | ds_base | cuda:1 | 0.7204 | 0.7875 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.7202 | 0.7875 |
| HPO_BILSTM | hpo | cuda:0 | 0.7194 | 0.7875 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.7171 | 0.7833 |
| DS_Transformer_B | ds_base | cuda:1 | 0.7164 | 0.7833 |
| DS_BiLSTM_A | ds_base | cuda:0 | 0.7154 | 0.7833 |
| CNN2DGRU | single | cuda:0 | 0.7153 | 0.7833 |
| HPO_CNN2DRNN | hpo | cuda:1 | 0.7148 | 0.7833 |
| DS_Mamba_B | ds_base | cuda:1 | 0.7141 | 0.7833 |
| BiLSTM | single | cuda:0 | 0.7136 | 0.7833 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.7123 | 0.7833 |
| Ensemble_TCN_Transformer | voting_ensemble | cuda:0 | 0.7115 | 0.7792 |
| DS_RNN_B | ds_base | cuda:1 | 0.7113 | 0.7792 |
| MoE_Mixed | soft_moe | cuda:0 | 0.7108 | 0.7792 |
| Ensemble_BiLSTM_LSTM | voting_ensemble | cuda:0 | 0.7081 | 0.7792 |
| DS_Transformer_A | ds_base | cuda:0 | 0.7079 | 0.7792 |
| Transformer | single | cuda:1 | 0.7073 | 0.7750 |
| MoE_4CNN | soft_moe | cuda:1 | 0.7067 | 0.7750 |
| LSTM | single | cuda:0 | 0.7048 | 0.7750 |
| DS_LSTM_A | ds_base | cuda:0 | 0.6955 | 0.7708 |
| HPO_RNN | hpo | cuda:0 | 0.6912 | 0.7667 |
| HPO_MAMBA | hpo | cuda:0 | 0.6908 | 0.7667 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-noise`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.6391 | 0.7273 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.5976 | 0.6962 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.5812 | 0.6829 |
| DS_GRU_A | ds_base | cuda:0 | 0.5706 | 0.6763 |
| HPO_GRU | hpo | cuda:0 | 0.5666 | 0.6741 |
| HPO_BILSTM | hpo | cuda:1 | 0.5634 | 0.6674 |
| MoE_4GRU | soft_moe | cuda:0 | 0.5625 | 0.6696 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.5614 | 0.6674 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.5613 | 0.6696 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.5609 | 0.6696 |
| HPO_MAMBA | hpo | cuda:0 | 0.5591 | 0.6674 |
| HPO_LSTM | hpo | cuda:1 | 0.5590 | 0.6674 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.5574 | 0.6674 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.5552 | 0.6652 |
| LSTM | single | cuda:0 | 0.5548 | 0.6652 |
| MoE_4TCN | soft_moe | cuda:0 | 0.5545 | 0.6652 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.5531 | 0.6630 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.5530 | 0.6630 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.5522 | 0.6630 |
| GRU | single | cuda:1 | 0.5519 | 0.6630 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.5518 | 0.6630 |
| HPO_TCN | hpo | cuda:1 | 0.5515 | 0.6630 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.5491 | 0.6608 |
| DS_Mamba_A | ds_base | cuda:0 | 0.5484 | 0.6608 |
| MoE_4CNN | soft_moe | cuda:1 | 0.5473 | 0.6585 |
| TCN | single | cuda:0 | 0.5469 | 0.6585 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.5455 | 0.6585 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.5425 | 0.6563 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.5423 | 0.6563 |
| DS_TCN_A | ds_base | cuda:1 | 0.5416 | 0.6563 |
| CNN2DLSTM | single | cuda:1 | 0.5403 | 0.6541 |
| DS_TCN_B | ds_base | cuda:0 | 0.5396 | 0.6541 |
| DS_RNN_B | ds_base | cuda:1 | 0.5392 | 0.6541 |
| CNN | single | cuda:1 | 0.5391 | 0.6541 |
| RNN | single | cuda:0 | 0.5390 | 0.6519 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.5367 | 0.6497 |
| DS_RNN_A | ds_base | cuda:0 | 0.5359 | 0.6497 |
| Mamba | single | cuda:0 | 0.5346 | 0.6497 |
| DS_CNN_B | ds_base | cuda:0 | 0.5343 | 0.6497 |
| DS_CNN_A | ds_base | cuda:1 | 0.5327 | 0.6475 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.5320 | 0.6452 |
| DS_Mamba_B | ds_base | cuda:1 | 0.5259 | 0.6408 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.5241 | 0.6408 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.5208 | 0.6341 |
| HPO_CNN | hpo | cuda:0 | 0.5157 | 0.6364 |
| MoE_Mixed | soft_moe | cuda:0 | 0.5031 | 0.6253 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-normal`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.6391 | 0.7273 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.5976 | 0.6962 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.5812 | 0.6829 |
| DS_GRU_A | ds_base | cuda:0 | 0.5706 | 0.6763 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.5667 | 0.6741 |
| HPO_GRU | hpo | cuda:0 | 0.5666 | 0.6741 |
| HPO_BILSTM | hpo | cuda:1 | 0.5634 | 0.6674 |
| MoE_4GRU | soft_moe | cuda:0 | 0.5625 | 0.6696 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.5614 | 0.6674 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.5613 | 0.6696 |
| HPO_MAMBA | hpo | cuda:0 | 0.5591 | 0.6674 |
| HPO_LSTM | hpo | cuda:1 | 0.5590 | 0.6674 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.5574 | 0.6674 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.5552 | 0.6652 |
| LSTM | single | cuda:0 | 0.5548 | 0.6652 |
| MoE_4TCN | soft_moe | cuda:0 | 0.5545 | 0.6652 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.5531 | 0.6630 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.5530 | 0.6630 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.5522 | 0.6630 |
| GRU | single | cuda:1 | 0.5519 | 0.6630 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.5518 | 0.6630 |
| HPO_TCN | hpo | cuda:1 | 0.5515 | 0.6630 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.5491 | 0.6608 |
| DS_Mamba_A | ds_base | cuda:0 | 0.5484 | 0.6608 |
| MoE_4CNN | soft_moe | cuda:1 | 0.5473 | 0.6585 |
| TCN | single | cuda:0 | 0.5469 | 0.6585 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.5455 | 0.6585 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.5425 | 0.6563 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.5423 | 0.6563 |
| DS_TCN_A | ds_base | cuda:1 | 0.5416 | 0.6563 |
| CNN2DLSTM | single | cuda:1 | 0.5403 | 0.6541 |
| DS_TCN_B | ds_base | cuda:0 | 0.5396 | 0.6541 |
| DS_RNN_B | ds_base | cuda:1 | 0.5392 | 0.6541 |
| CNN | single | cuda:1 | 0.5391 | 0.6541 |
| RNN | single | cuda:0 | 0.5390 | 0.6519 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.5367 | 0.6497 |
| DS_RNN_A | ds_base | cuda:0 | 0.5359 | 0.6497 |
| Mamba | single | cuda:0 | 0.5346 | 0.6497 |
| DS_CNN_B | ds_base | cuda:0 | 0.5343 | 0.6497 |
| DS_CNN_A | ds_base | cuda:1 | 0.5327 | 0.6475 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.5320 | 0.6452 |
| DS_Mamba_B | ds_base | cuda:1 | 0.5259 | 0.6408 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.5208 | 0.6341 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.5183 | 0.6364 |
| HPO_CNN | hpo | cuda:0 | 0.5157 | 0.6364 |
| MoE_Mixed | soft_moe | cuda:0 | 0.5031 | 0.6253 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-normal`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| DS_CNN_A | ds_base | cuda:1 | 0.7677 | 0.8250 |
| DS_RNN_A | ds_base | cuda:0 | 0.7674 | 0.8250 |
| Mamba | single | cuda:0 | 0.7583 | 0.8167 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.7575 | 0.8167 |
| MoE_4TCN | soft_moe | cuda:0 | 0.7550 | 0.8125 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.7545 | 0.8125 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.7533 | 0.8125 |
| MoE_4GRU | soft_moe | cuda:0 | 0.7522 | 0.8125 |
| HPO_LSTM | hpo | cuda:1 | 0.7514 | 0.8125 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.7489 | 0.8083 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.7487 | 0.8083 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.7482 | 0.8083 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.7478 | 0.8083 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7445 | 0.8042 |
| DS_CNN_B | ds_base | cuda:0 | 0.7421 | 0.8042 |
| CNN | single | cuda:1 | 0.7413 | 0.8042 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.7407 | 0.8042 |
| HPO_TCN | hpo | cuda:1 | 0.7396 | 0.8042 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.7378 | 0.8000 |
| DS_Mamba_A | ds_base | cuda:0 | 0.7362 | 0.8000 |
| TCN | single | cuda:0 | 0.7347 | 0.7958 |
| GRU | single | cuda:1 | 0.7323 | 0.7958 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.7306 | 0.7958 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.7297 | 0.7958 |
| DS_TCN_A | ds_base | cuda:1 | 0.7295 | 0.7917 |
| RNN | single | cuda:0 | 0.7290 | 0.7958 |
| HPO_CNN | hpo | cuda:0 | 0.7288 | 0.7958 |
| DS_GRU_A | ds_base | cuda:0 | 0.7285 | 0.7958 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7273 | 0.7917 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.7266 | 0.7917 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.7260 | 0.7917 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.7251 | 0.7917 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.7238 | 0.7917 |
| CNN2DLSTM | single | cuda:1 | 0.7214 | 0.7875 |
| DS_TCN_B | ds_base | cuda:0 | 0.7213 | 0.7875 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.7202 | 0.7875 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.7171 | 0.7833 |
| HPO_BILSTM | hpo | cuda:1 | 0.7152 | 0.7833 |
| DS_Mamba_B | ds_base | cuda:1 | 0.7141 | 0.7833 |
| HPO_GRU | hpo | cuda:0 | 0.7126 | 0.7833 |
| DS_RNN_B | ds_base | cuda:1 | 0.7113 | 0.7792 |
| MoE_Mixed | soft_moe | cuda:0 | 0.7108 | 0.7792 |
| HPO_MAMBA | hpo | cuda:0 | 0.7077 | 0.7792 |
| MoE_4CNN | soft_moe | cuda:1 | 0.7067 | 0.7750 |
| LSTM | single | cuda:0 | 0.7048 | 0.7750 |
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.7037 | 0.7750 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-noise`

## Deep Learning Results (seed=42)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0', 'cuda:1']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| DS_CNN_A | ds_base | cuda:1 | 0.7677 | 0.8250 |
| DS_RNN_A | ds_base | cuda:0 | 0.7674 | 0.8250 |
| Mamba | single | cuda:0 | 0.7583 | 0.8167 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.7575 | 0.8167 |
| MoE_4TCN | soft_moe | cuda:0 | 0.7550 | 0.8125 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.7545 | 0.8125 |
| MoE_Mixed2 | soft_moe | cuda:1 | 0.7533 | 0.8125 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.7527 | 0.8125 |
| MoE_4GRU | soft_moe | cuda:0 | 0.7522 | 0.8125 |
| HPO_LSTM | hpo | cuda:1 | 0.7514 | 0.8125 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.7489 | 0.8083 |
| MoE_2TCN2GRU | soft_moe | cuda:1 | 0.7487 | 0.8083 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.7473 | 0.8083 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7445 | 0.8042 |
| DS_CNN_B | ds_base | cuda:0 | 0.7421 | 0.8042 |
| CNN | single | cuda:1 | 0.7413 | 0.8042 |
| MoE_4LSTM | soft_moe | cuda:1 | 0.7407 | 0.8042 |
| HPO_TCN | hpo | cuda:1 | 0.7396 | 0.8042 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.7378 | 0.8000 |
| DS_Mamba_A | ds_base | cuda:0 | 0.7362 | 0.8000 |
| TCN | single | cuda:0 | 0.7347 | 0.7958 |
| GRU | single | cuda:1 | 0.7323 | 0.7958 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.7306 | 0.7958 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.7297 | 0.7958 |
| DS_TCN_A | ds_base | cuda:1 | 0.7295 | 0.7917 |
| RNN | single | cuda:0 | 0.7290 | 0.7958 |
| HPO_CNN | hpo | cuda:0 | 0.7288 | 0.7958 |
| DS_GRU_A | ds_base | cuda:0 | 0.7285 | 0.7958 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7273 | 0.7917 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.7266 | 0.7917 |
| DS_CNN2DRNN_B | ds_base | cuda:1 | 0.7260 | 0.7917 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.7251 | 0.7917 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.7238 | 0.7917 |
| CNN2DLSTM | single | cuda:1 | 0.7214 | 0.7875 |
| DS_TCN_B | ds_base | cuda:0 | 0.7213 | 0.7875 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.7202 | 0.7875 |
| MoE_8Mixed | soft_moe | cuda:1 | 0.7171 | 0.7833 |
| HPO_BILSTM | hpo | cuda:1 | 0.7152 | 0.7833 |
| DS_Mamba_B | ds_base | cuda:1 | 0.7141 | 0.7833 |
| HPO_GRU | hpo | cuda:0 | 0.7126 | 0.7833 |
| DS_RNN_B | ds_base | cuda:1 | 0.7113 | 0.7792 |
| MoE_Mixed | soft_moe | cuda:0 | 0.7108 | 0.7792 |
| HPO_MAMBA | hpo | cuda:0 | 0.7077 | 0.7792 |
| MoE_4CNN | soft_moe | cuda:1 | 0.7067 | 0.7750 |
| LSTM | single | cuda:0 | 0.7048 | 0.7750 |
| MetaFusion_GRU | meta_fusion | cuda:1 | 0.7037 | 0.7750 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-noise`

## Deep Learning Results (seed=3)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| RNN | single | cuda:0 | 0.8808 | 0.9062 |
| HPO_BILSTM | hpo | cuda:0 | 0.8784 | 0.9062 |
| MoE_4GRU | soft_moe | cuda:0 | 0.8784 | 0.9062 |
| MoE_Mixed | soft_moe | cuda:0 | 0.8784 | 0.9062 |
| HPO_CNN | hpo | cuda:0 | 0.8761 | 0.9062 |
| MoE_4CNN | soft_moe | cuda:0 | 0.8761 | 0.9062 |
| HPO_GRU | hpo | cuda:0 | 0.8444 | 0.8750 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.8388 | 0.8750 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.8388 | 0.8750 |
| DS_RNN_A | ds_base | cuda:0 | 0.8388 | 0.8750 |
| GRU | single | cuda:0 | 0.8377 | 0.8750 |
| HPO_LSTM | hpo | cuda:0 | 0.8377 | 0.8750 |
| MoE_Mixed2 | soft_moe | cuda:0 | 0.8377 | 0.8750 |
| DS_GRU_A | ds_base | cuda:0 | 0.8366 | 0.8750 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.8344 | 0.8750 |
| MoE_2TCN2GRU | soft_moe | cuda:0 | 0.8054 | 0.8438 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.8054 | 0.8438 |
| CNN | single | cuda:0 | 0.8043 | 0.8438 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.8043 | 0.8438 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.8033 | 0.8438 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.8022 | 0.8438 |
| MoE_4LSTM | soft_moe | cuda:0 | 0.8022 | 0.8438 |
| DS_CNN_B | ds_base | cuda:0 | 0.7990 | 0.8438 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7948 | 0.8438 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.7948 | 0.8438 |
| CNN2DLSTM | single | cuda:0 | 0.7917 | 0.8438 |
| DS_TCN_A | ds_base | cuda:0 | 0.7630 | 0.8125 |
| HPO_TCN | hpo | cuda:0 | 0.7589 | 0.8125 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.7589 | 0.8125 |
| Mamba | single | cuda:0 | 0.7549 | 0.8125 |
| HPO_MAMBA | hpo | cuda:0 | 0.7549 | 0.8125 |
| DS_RNN_B | ds_base | cuda:0 | 0.7529 | 0.8125 |
| LSTM | single | cuda:0 | 0.7520 | 0.8125 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.7520 | 0.8125 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7520 | 0.8125 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.7520 | 0.8125 |
| DS_Mamba_B | ds_base | cuda:0 | 0.7510 | 0.8125 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.7265 | 0.7812 |
| DS_CNN2DRNN_B | ds_base | cuda:0 | 0.7246 | 0.7812 |
| MoE_4TCN | soft_moe | cuda:0 | 0.7149 | 0.7812 |
| MetaFusion_GRU | meta_fusion | cuda:0 | 0.7093 | 0.7812 |
| DS_TCN_B | ds_base | cuda:0 | 0.6819 | 0.7500 |
| MoE_8Mixed | soft_moe | cuda:0 | 0.6728 | 0.7500 |
| DS_CNN_A | ds_base | cuda:0 | 0.6325 | 0.7188 |
| DS_Mamba_A | ds_base | cuda:0 | 0.6308 | 0.7188 |
| TCN | single | cuda:0 | 0.4967 | 0.5938 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-pure`

## Deep Learning Results (seed=5)

### Setup
- Models: RNN, GRU, LSTM (attention), CNN (multi-scale residual), TCN, Transformer
- Ensembles: 6 voting combos + stacking meta-learner
- NAS: Optuna joint arch + HP search (`optuna_dl.db`)
- GPUs: ['cuda:0']
- Dataset: 4 classes, seq_len=10, in_features=1
- Regularisation: L1 λ=1e-05, AdamW wd=0.0001, dropout=0.3
- Primary metric: Matthews Correlation Coefficient (MCC)
- CV folds: 5

### Results

| Model | Type | Device | Test MCC | Test Acc |
|-------|------|--------|----------|----------|
| CNN | single | cuda:0 | 0.9215 | 0.9375 |
| MoE_Mixed2 | soft_moe | cuda:0 | 0.8831 | 0.9062 |
| DS_Mamba_A | ds_base | cuda:0 | 0.8808 | 0.9062 |
| Ensemble_CNN_TCN | voting_ensemble | cuda:0 | 0.8761 | 0.9062 |
| Stacking_All | stacking_ensemble | cuda:0 | 0.8761 | 0.9062 |
| DS_RNN_A | ds_base | cuda:0 | 0.8761 | 0.9062 |
| MoE_4TCN | soft_moe | cuda:0 | 0.8455 | 0.8750 |
| MoE_4GRU | soft_moe | cuda:0 | 0.8388 | 0.8750 |
| TCN | single | cuda:0 | 0.8366 | 0.8750 |
| Ensemble_LSTM_CNN | voting_ensemble | cuda:0 | 0.8366 | 0.8750 |
| MoE_2CNN2TCN | soft_moe | cuda:0 | 0.8355 | 0.8750 |
| DS_RNN_B | ds_base | cuda:0 | 0.8344 | 0.8750 |
| DS_GRU_A | ds_base | cuda:0 | 0.8344 | 0.8750 |
| DS_CNN2DRNN_B | ds_base | cuda:0 | 0.8344 | 0.8750 |
| HPO_TRANSFORMER | hpo | cuda:0 | 0.8011 | 0.8438 |
| LSTM | single | cuda:0 | 0.7990 | 0.8438 |
| HPO_CNN | hpo | cuda:0 | 0.7990 | 0.8438 |
| DeepStackEnsemble | deep_stack | cuda:0 | 0.7990 | 0.8438 |
| Mamba | single | cuda:0 | 0.7969 | 0.8438 |
| MoE_8Mixed | soft_moe | cuda:0 | 0.7969 | 0.8438 |
| HPO_TCN | hpo | cuda:0 | 0.7948 | 0.8438 |
| DS_CNN_A | ds_base | cuda:0 | 0.7948 | 0.8438 |
| Ensemble_Mamba_LSTM | voting_ensemble | cuda:0 | 0.7927 | 0.8438 |
| DS_Mamba_B | ds_base | cuda:0 | 0.7927 | 0.8438 |
| RNN | single | cuda:0 | 0.7917 | 0.8438 |
| Ensemble_RNN_GRU | voting_ensemble | cuda:0 | 0.7917 | 0.8438 |
| DS_TCN_B | ds_base | cuda:0 | 0.7610 | 0.8125 |
| DS_TCN_A | ds_base | cuda:0 | 0.7589 | 0.8125 |
| MetaFusion_LSTM | meta_fusion | cuda:0 | 0.7529 | 0.8125 |
| MetaFusion_GRU | meta_fusion | cuda:0 | 0.7529 | 0.8125 |
| HPO_MAMBA | hpo | cuda:0 | 0.7529 | 0.8125 |
| MoE_2GRU2LSTM | soft_moe | cuda:0 | 0.7529 | 0.8125 |
| Ensemble_RNN_GRU_LSTM | voting_ensemble | cuda:0 | 0.7510 | 0.8125 |
| Ensemble_All | voting_ensemble | cuda:0 | 0.7510 | 0.8125 |
| MoE_4LSTM | soft_moe | cuda:0 | 0.7510 | 0.8125 |
| MoE_Mixed | soft_moe | cuda:0 | 0.7510 | 0.8125 |
| CNN2DLSTM | single | cuda:0 | 0.7265 | 0.7812 |
| MoE_4CNN | soft_moe | cuda:0 | 0.7206 | 0.7812 |
| DS_CNN_B | ds_base | cuda:0 | 0.7130 | 0.7812 |
| Ensemble_GRU_LSTM | voting_ensemble | cuda:0 | 0.7121 | 0.7812 |
| OptunaNet_NAS | optuna_nas | cuda:0 | 0.7121 | 0.7812 |
| HPO_BILSTM | hpo | cuda:0 | 0.7111 | 0.7812 |
| GRU | single | cuda:0 | 0.7093 | 0.7812 |
| HPO_LSTM | hpo | cuda:0 | 0.6719 | 0.7500 |
| HPO_GRU | hpo | cuda:0 | 0.5903 | 0.6875 |
| MoE_2TCN2GRU | soft_moe | cuda:0 | 0.5903 | 0.6875 |

### Notes
- Single models trained in parallel (one per GPU, round-robin)
- HPO studies run in parallel across GPUs
- All models saved under `models/dl/`
- Confusion matrices + curves in `plots/dl/`
- WandB project: `inMotion-dl-pure`
