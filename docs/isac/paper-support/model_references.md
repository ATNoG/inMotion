# ISAC 2026 — Model References

Compiled references supporting every model cited in the paper. Each entry gives
the canonical citation, the arXiv/URL, and what the model contributes to this work.

---

## 1. JEPA family (self-supervised world models)

| Model | Reference | Where it's used |
|---|---|---|
| **T-JEPA** | Thimonier, Popineau, Rimmel, Doan, Daniel (2024). *T-JEPA: A Joint-Embedding Predictive Architecture for Tabular Data*. arXiv:2410.05016. Code: https://github.com/jose-melo/t-jepa | `t_jepa` — our 4-channel / 18-channel tabular-JEPA encoder. See `dl/models/t_jepa.py` |
| **I-JEPA** (foundation) | Assran, Duval, Misra, Bojanowski, Ballas, Gondal, Avent, Mazare (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243 | Conceptual basis of all JEPA variants (context/target encoder + predictor) |
| **TS-JEPA** | Time-Series Joint-Embedding Predictive Architecture (2024). arXiv:2509.25449 | `ts_jepa` — Conv1D-patched, contiguous temporal-block masking over multi-channel input. See `dl/models/ts_jepa.py` |
| **LeJEPA** | Balestriero & LeCun (2025). *LeJEPA: Joint-Embedding Predictive Architecture with Sliced Gaussian Regularization*. arXiv:2511.08544. Code: https://github.com/galilai-group/lejepa | `lejepa` — end-to-end next-step latent prediction with SIGReg anti-collapse. See `dl/models/lejepa.py` |
| **CF-JEPA** | *CF-JEPA: Mask-free multi-horizon forward prediction for time series*. arXiv:2606.07031 | `cf_jepa` — mask-free multi-horizon forward prediction (random context crop → predict future latents). See `dl/models/cf_jepa.py` |
| **SIGReg** | Same LeJEPA paper (Balestriero & LeCun 2025) — ECF-based Gaussian regularizer | `sigreg` classifier and the anti-collapse term in all JEPA models. See `dl/sigreg.py` |
| **LeWorldModel** | le-wm: https://github.com/lucas-maes/le-wm ; arXiv:2603.19312 | Ancillary source used while exploring JEPA-style world models |
| **RDMReg / LpJEPA** | *Rectified LpJEPA* (2026). arXiv:2602.01456 | Sparse non-negative alternative regularizer implemented in `dl/sigreg.py` (RDMReg), used as `--sigreg-mode rdm` |

## 2. State-space models

| Model | Reference | Where it's used |
|---|---|---|
| **Mamba** | Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752 | `mamba` base model (deep-stack base, HPO_MAMBA) |
| **Mamba-3 family** | Gu, Parnichkun, et al. (2026). arXiv:2606.xxxxx (mamba3) | `mamba3_cnn`, `mamba3_tcn`, `mamba3_transformer`, `mamba3_multiview` — four fusion variants. See `dl/models/mamba3_*.py` |

## 3. Classic deep-sequence models

| Model | Reference | Where it's used |
|---|---|---|
| **TCN** | Bai, Kolter, Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271 | `tcn` classifier, deep-stack TCN bases |
| **RNN/GRU/LSTM/BiLSTM** | Hochreiter & Schmidhuber (1997) *LSTM*; Cho et al. (2014) *GRU* | `gru`, `lstm`, `bilstm`, HPO_GRU/LSTM/RNN, deep-stack bases |
| **CNN2DRNN** | — (local architecture) | `cnn2d_rnn`, deep-stack CNN2DRNN base |
| **SoftMoE** | Puigcerver et al. (2023). *From Sparse to Soft Mixtures of Experts*. arXiv:2308.00951 | `MoE_4TCN`, `MoE_4LSTM`, `MoE_Mixed2` |
| **Autoformer** | Wu et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. arXiv:2106.13008 | `Autoformer` variant (paper architecture table) |
| **DeepStackEnsemble** | Deep stacking: Deng & Yu (2011) *Deep Stacking Networks*; ensemble flavor adapted here | `deepstack` — 10 bases → per-class Level2Nets → DeepMetaMLP. See `dl/models/deep_stack.py` |

## 4. Classical ML baselines

| Model | Reference | Where it's used |
|---|---|---|
| **ROCKET** | Dempster, Petitjean, Webb (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels*. Data Mining and Knowledge Discovery 34(5):1454–1495 | `rocket` — 512 random kernels + logistic regression. See `mega_ensemble.py:rocket_features` |
| **catch22** | Lubba et al. (2019). *catch22: CAnonical Time-series CHaracteristics*. Data Mining and Knowledge Discovery 33:1821–1852 | `catch22` — 22 canonical time-series features + logistic regression. See `mega_ensemble.py:catch22_features` |
| **TabPFN** | Hollmann et al. (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*. arXiv:2207.01848 | `tabpfn` prior-fitting transformer member (optional, `--no-tabpfn` default off) |
| **kNN / Logistic Regression / GP / CatBoost** | standard; GP: Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*; CatBoost: Prokhorenkova et al. (2018). arXiv:1706.09516 | classical members in `mega_ensemble.py` |

## 5. Ensembling

| Method | Reference | Where it's used |
|---|---|---|
| **Caruana ensemble selection** | Caruana, Niculescu-Mizil, Crew, Ksikes (2004). *Ensemble Selection from Libraries of Models*. ICML 2004 | `_caruana_selection` / `_bagged_caruana` combiner |
| **Logistic-regression stacking** | Wolpert (1992). *Stacked Generalization*. Neural Networks 5(2):241–259 | `lr_stack` combiner — multinomial LR over concatenated calibrated probabilities |
| **Temperature calibration** | Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks*. ICML 2017 | `fit_temperature` — per-member scalar temperature fit on validation NLL |
| **Entropy-weighted averaging** | (standard uncertainty weighting) | `_entropy_weighted` combiner |
| **MCC metric** | Chicco & Jurman (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation*. BMC Genomics 21:6 | primary metric throughout |

## 6. Supporting / related

| Topic | Reference |
|---|---|
| Prior EUCNC paper (data collection protocol) | Ribeiro et al. (2026), EUCNC — `\cite{ribeiro2026eucnc}` in the paper |
| Vision-JEPA predecessor | Assran et al. (2023) I-JEPA (above) |
