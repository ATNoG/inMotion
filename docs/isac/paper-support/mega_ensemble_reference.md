# ISAC 2026 — Mega-Ensemble Reference

Full description of the mega-ensemble: member roster, feature pipelines,
calibration, combiners, and the exact ensembles named in the paper.

---

## 1. Member roster (21 DL members)

| # | Member | Family | Features | Test MCC (icaisf) | Params (M) |
|---|---|---|---|---|---|
| 1 | lejepa | JEPA | plain(4) | 0.8850 | 0.566 |
| 2 | t_jepa | JEPA | rich(18) | 0.8694 | 0.633 |
| 3 | ts_jepa | JEPA | rich(18) | 0.8785 | 2.153 |
| 4 | cf_jepa | JEPA | rich(18) | 0.8661 | 1.232 |
| 5 | sigreg | SIGReg | plain(4) | 0.8858 | 0.697 |
| 6 | sigreg_s3 | SIGReg (seed 3) | plain(4) | 0.8706 | 0.697 |
| 7 | sigreg_s5 | SIGReg (seed 5) | plain(4) | 0.8795 | 0.697 |
| 8 | mamba3_cnn | Mamba-3 | plain(4) | 0.8567 | 0.945 |
| 9 | mamba3_tcn | Mamba-3 | plain(4) | 0.8844 | 0.843 |
| 10 | mamba3_transformer | Mamba-3 | plain(4) | 0.8581 | 0.842 |
| 11 | mamba3_multiview | Mamba-3 | plain(4) | 0.8640 | 1.701 |
| 12 | hpo_gru | HPO DL | plain(4) | 0.5306 | 4.008 |
| 13 | hpo_lstm | HPO DL | plain(4) | 0.6092 | 13.110 |
| 14 | hpo_cnn | HPO DL | plain(4) | 0.5785 | 0.081 |
| 15 | hpo_mamba | HPO DL | plain(4) | 0.6049 | 0.278 |
| 16 | deepstack | DeepStack | plain(4) | 0.8810 | 43.576 |
| 17 | cnn | classic DL | plain(4) | 0.6201* | 0.317 |
| 18 | tcn | classic DL | plain(4) | 0.5545 | 2.180 |
| 19 | gru | classic DL | plain(4) | 0.3360 | 0.039 |
| 20 | lstm | classic DL | plain(4) | 0.5547 | 0.052 |
| 21 | bilstm | classic DL | plain(4) | 0.5622 | 0.136 |

\*cnn on dataset.csv; on icaisf see MASTER_RESULTS §5 (0.62 range).
Full per-member table: `docs/isac/results/MASTER_RESULTS.md` §5.

**Key design rule**: JEPA-family members that were trained with rich features
consume the 18-channel pipeline (`rich=True`); every other member consumes the
4-channel pipeline. The feature pipeline per member is fixed by its checkpoint's
training config — never mixed.

## 2. Calibration

Every member's softmax probabilities are temperature-calibrated before any
combination:

- Pseudo-logits: `log(p)`, then `softmax(logits / T)`.
- T fit per member on **validation** NLL (gradient descent, 100 iters, init 1.0,
  floor 0.05).
- Val and test probabilities both rescaled with the same T.

## 3. Combiners (all compared in `mega_ensemble.py`)

| Combiner | Construction | Test MCC (all-21, icaisf) |
|---|---|---|
| **LR-stack (winner)** | Multinomial LogisticRegression on the concatenated calibrated probs `(n_members×4)`; C swept [0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]; best C on val MCC; max_iter=5000 | **0.8873** (C=0.005) |
| Entropy-weighted | Weight each member by inverse entropy of its val probs | 0.8873 |
| Plain average | Uniform mean of calibrated probs | 0.8814 |
| Bagged-Caruana | 20 bags × 50 greedy selection steps on 60% val subsamples, CE metric, weight = avg pick count | 0.8706 |
| Rank average | Rank-transform probs then average | −0.316 (unusable) |

**Winner reported in the paper: LR-stack, test MCC 0.8873** on the 21-member
ensemble (icaisf).

## 4. Named ensembles in the paper (must cite exact member sets)

| Name | Members | Combiner | Test MCC |
|---|---|---|---|
| E1 "all 21" | members 1–21 above | LR-stack (C=0.005), calibrated | 0.8873 (icaisf) |
| E2 size-frontier | logreg → +lejepa → +sigreg | LR-stack (greedy budget) | **0.8904** @ 1.26 M |
| E3 "16 members" | lejepa, t_jepa, ts_jepa, cf_jepa, sigreg, mamba3_cnn/tcn/transformer/multiview, deepstack, cnn, tcn, hpo_gru, hpo_lstm, hpo_cnn, hpo_mamba | LR-stack | **0.9232** (dataset.csv, C=0.5) / 0.9563 (noise, C=0.1) / 0.9179 (pure, C=0.005) |
| E4 world-only | lejepa, t_jepa, ts_jepa, cf_jepa, sigreg, sigreg_s3, sigreg_s5 | equal-weight mean | 0.8872 (val 0.9063) |
| E5 world+best-DL | E4 + deepstack, mamba3_tcn | equal-weight mean | 0.8872 (val 0.9178) |
| E6 best-world | sigreg, lejepa, ts_jepa | equal-weight mean | 0.8784 (val 0.9183) |
| E7 best-world+best-DL | sigreg, lejepa, ts_jepa + deepstack, mamba3_tcn | equal-weight mean | 0.8872 (val 0.9069) |
| E8 greedy-selected | 16 members chosen greedily on val MCC (see MASTER_RESULTS §4.2 for the ordered list) | equal-weight mean | 0.8873 (val 0.9240) |

**E1 vs E3 caveat**: E1 (all 21 incl. weak recurrents) scores 0.8873 on icaisf;
E3 (16 strong, non-redundant members) scores 0.9232 on dataset.csv. The paper
must state which member set produced each number.

## 5. Size frontier (deployment tradeoff)

| Step | Members | Total params | Test MCC |
|---|---|---|---|
| 1 | logreg | 0.0 M | 0.5305 |
| 2 | + lejepa | 0.57 M | 0.8694 |
| 3 | + sigreg | 1.26 M | **0.8904** |

Greedy by MCC-per-added-param (equal-weight mean of calibrated probs). This
0.8904 is the highest honest test MCC on icaisf — a 3-member subset beats the
21-member LR-stack (0.8873) at 1/30th the size. Note: the logreg member here is
the classical LR on raw columns, not the LR-stack combiner.

## 6. Known caveats (must not be papered over)

- **hpo_lstm (13.1 M) and hpo_gru (4.0 M) underperform** on icaisf (0.61/0.53)
  despite big size — they were HPO'd on the augmented dataset and do not
  transfer. They add diversity but drag the equal-weight ensembles; the LR-stack
  down-weights them (C=0.005).
- **DeepStack** is 43.6 M/10 bases in the actual checkpoint, not the 1.2 M/18
  bases in the older paper table — the paper must cite the measured checkpoint.
- **Rank-average is unusable** (−0.316) — reported only for completeness.
- The full-after-hpo sigreg (trained on augmented3) does **not** transfer to
  icaisf (0.0185) — the ensemble uses the icaisf-trained backup sigreg instead.
  This is a dataset-mismatch, not an architecture failure.
