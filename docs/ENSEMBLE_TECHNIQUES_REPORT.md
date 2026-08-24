# Ensemble Techniques for Pushing MCC 0.82 → 0.90
**Research report — 4-class, 10-timestep RSSI classification, ~2,000–10,000 training samples.**
*Goal: combine heterogeneous families (transformers/CNNs/JEPA, trees, kernels) at the ensemble level.*

---

## Context assumptions

| Asset | MCC |
|---|---|
| Caruana greedy selection (5 DL models: lejepa, t_jepa, ts_jepa, sigreg, mamba3_cnn) | 0.815 |
| DeepStack (18 base DL → per-class Level2Nets → residual DeepMetaMLP) | 0.859 |
| Gaussian Process (pure data) | 0.756 |
| CatBoost (pure data) | 0.770 |
| **KNN (pure data)** | **0.922** |
| Target | ≥ 0.90 |

## Headline synthesis (read this first)

The single largest untapped signal is that **KNN on pure data already scores 0.922 while the deep stack sits at 0.859** — the classical models (KNN above all, then CatBoost and GP once calibrated) are *not* members of the current ensemble at all. The fastest path to 0.9 is a pipeline that:

1. **Calibrates every member into a common probability space** (per-model temperature scaling) so heterogeneous outputs are comparable,
2. **Feeds out-of-fold (OOF) probabilities from ALL families — DL logits, KNN, CatBoost, GP, TabPFN — into one regularized meta-learner** (this is the single biggest lever),
3. **Selects members with an upgraded Caruana procedure** (proper scoring rule + bagged selection + top-N init), and
4. **Uses rank averaging as a robust fallback blend** that is insensitive to score-scale mismatches.

Expected trajectory: 0.859 → ~0.88–0.91 (deep stack + KNN/CatBoost/GP OOFs + TabPFN + calibration + selection). One strategic caveat: per `ROADMAP_TO_90.md`, the JEPA models have only had ~3 pretraining epochs — ensembling cannot fully compensate for under-trained bases. Full JEPA pretraining (Phase 1 of the roadmap) should be run *before* final ensemble construction, because it changes the base-model quality that every technique below depends on.

---

## Ranked techniques (14)

### 1. Heterogeneous OOF stacking (stacked generalization across families)
**Sources:** [Stacked Generalization: A Comprehensive Guide for 2025](https://www.shadecoder.com/topics/stacked-generalization-a-comprehensive-guide-for-2025) · [kaggle-tabular-forge stacking.md](https://github.com/VectorPeak/kaggle-tabular-forge/blob/main/docs/stacking.md) · [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

**Core idea:** Train every base model under the *same* k-fold split, collect OOF probability vectors, and fit a simple, strongly regularized meta-learner on those OOF vectors. This is Wolpert's stacked generalization and it is exactly how Kaggle winners fuse tree + NN + linear families. For you the key move is putting KNN (0.922), CatBoost (0.77) and GP (0.756) OOFs into the same meta-feature matrix as the 5 DL models — the meta-learner learns *when* to trust KNN vs. the DL models per class, which is where the 0.82 → 0.9 gap lives.

**Recipe:**
```
F = StratifiedKFold(10, shuffle=True, seed=42)
models = [lejepa, t_jepa, ts_jepa, sigreg, mamba3_cnn, KNN, CatBoost, GP, TabPFN]
for m in models:
    for tr, va in F.split(X, y):
        m.fit(X[tr], y[tr])
        OOF[va, m] = flatten(m.predict_proba(X[va]))     # 4 cols per model
    m.fit(X, y)                                           # refit on 100% for test
    TEST[m] = flatten(m.predict_proba(X_test))
# optional meta-features: raw X, DL embeddings, per-sample entropy of each member
Z  = hstack([OOF[m]  for m in models])
Zt = hstack([TEST[m] for m in models])
meta = LogisticRegression(C=0.1, penalty='l2')            # simple > flexible on 2-10k rows
meta.fit(Z, y)
y_pred = meta.predict(Zt)
```

**Expected benefit:** Largest single lever — +0.02 to +0.05 MCC over the best single member, because it harvests the KNN-vs-DL complementarity directly.

**Pitfalls / when NOT to use:** Leakage if OOF isn't used (never train the meta on in-sample predictions). With 2–10k samples the meta-learner overfits fast — use regularized LR or a tiny MLP with dropout, drop highly correlated meta-features, and nested-CV to check the meta. Trees as meta-learner overfit OOF predictions; avoid.

---

### 2. Calibration into a common probability space (per-model temperature scaling)
**Sources:** [Guo et al., ICML 2017 — On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) · [On Joint Regularization and Calibration in Deep Ensembles (arXiv 2511.04160, Nov 2025)](https://arxiv.org/abs/2511.04160) · [GETS: Ensemble Temperature Scaling (arXiv 2410.09570)](https://arxiv.org/abs/2410.09570) · [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html)

**Core idea:** Transformers and CNNs emit logits on wildly different scales, and KNN/trees emit scores that aren't probabilities at all. Averaging or stacking those directly is dominated by scale, not signal. Fitting a per-model temperature `T_m` (one scalar, minimizing NLL on a small held-out set) maps every member into a comparable, calibrated probability space before any combination. The 2025 paper shows an *ensemble optimality gap*: tuning temperature **jointly on the ensemble's averaged output** beats tuning members individually — and a 1–5% validation split is enough; bigger splits hurt by starving training.

**Recipe:**
```
X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.05, stratify=y)
for m in models:
    m.fit(X_tr, y_tr)
    T_m = minimize(lambda T: NLL(softmax(m.logits(X_cal)/T), y_cal))   # 1-D scipy optimize
    P_cal[m] = softmax(m.logits(X_cal)/T_m)
# Joint variant (preferred per 2025 results):
T_joint = minimize(lambda T: NLL(mean_m(softmax(m.logits(X_cal)/T)), y_cal))
P_ens = mean_m(softmax(m.logits(X)/T_joint))
```

**Expected benefit:** Prerequisite for techniques #1, #4, #6, #7. Fixes the scale mismatch that silently poisons heterogeneous blending; typically a small MCC bump (a few thousandths to ~0.01) but large calibration/robustness gains.

**Pitfalls / when NOT to use:** Tiny calibration sets give unstable `T` (keep val at 1–5%, or use the *overlapping holdout* idea from 2511.04160). Don't calibrate members individually *then* average if you are tuning the ensemble — joint scaling is better. Logit averaging raises confidence: if your models are already overconfident (common for transformers), calibration-then-average can increase calibration error.

---

### 3. Upgraded Caruana ensemble selection (proper scoring rule + bagged selection + top-N init)
**Sources:** [Caruana et al., ICML 2004 — Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) · [caruana_stacker.py gist (bagged selection reference implementation)](https://gist.github.com/bbstats/72e3741c28c39a4fc497b04409c4fb66) · [Selection criterion analysis (cross-entropy vs accuracy)](https://github.com/petteriTeikari/vascadia/issues/894) · [Getting the Most Out of Ensemble Selection](https://www.cs.cornell.edu/~mmunson/publications/docs/enssel_most.pdf)

**Core idea:** You already run greedy forward selection with replacement over 5 models (0.815). The published safeguards you can still add: (a) select on OOF **cross-entropy (a proper scoring rule)** instead of accuracy — Caruana-style selection with CE improves monotonically with ensemble size, while accuracy-based selection plateaus; (b) **bagged selection** — repeat the greedy procedure on random 50% subsets of the model library and average the pick counts into weights; (c) **initialize each bag with the top-N individually best models**. Also: your library of 5 is far too small — after #1, run selection over 15–40 members (5 DL × 5 seeds, KNN, CatBoost, GP, TabPFN, snapshots).

**Recipe:**
```
OOF = collect_probabilities(all_members)          # n × M × 4 (M ≥ 15)
weights = zeros(M)
for b in range(20):
    lib  = sample(M, frac=0.5, replace=False)
    picks = top_N_by_logloss(lib)                  # init safeguard
    S    = mean(OOF[picks], axis=0)
    for step in range(200):
        best = argmin_m logloss(y, mean([S, OOF[m]]))
        if logloss not improved: break
        picks.append(best); S = mean(OOF[picks])
    weights[picks] += 1
weights /= weights.sum()
final_proba = sum_m weights[m] * TEST[m]
```

**Expected benefit:** +0.005 to +0.015 MCC over your current 0.815, plus sparse robust weights that stop any single model from dominating. This is also the exact machinery behind TabPFN's PHE ensemble (Nature 2025).

**Pitfalls / when NOT to use:** Selection on a 2k-sample validation set overfits — bagged selection is the mitigation, don't skip it. With only 5 members there is nothing to select; expand the library first. If all members are highly correlated (JEPA family), selection gains shrink.

---

### 4. Rank averaging / rank-space blending
**Sources:** [Want to Win at Kaggle? Pay Attention to Your Ensembles](https://freedom251.com/want-to-win-at-kaggle-pay-attention-to-your-ensembles/) · [Kaggle Winning Solutions (collection)](https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions) · [Rank aggregation vs averaging (comparison)](https://blog.truegeometry.com/api/exploreHTML/229560297cfdf59361466e4c18445cd3.exploreHTML)

**Core idea:** Replace raw scores with per-class ranks before averaging. Rank averaging is scale-invariant: KNN's raw distances, CatBoost's margin scores, and transformer logits all become comparable, so a weak-calibrated but strong-ranking model (KNN) contributes its full signal. This is a time-honored Kaggle blend trick that shines exactly when members come from different model families.

**Recipe:**
```
R_m = argsort(argsort(-P_m, axis=1), axis=1)     # rank of each class within sample
R   = mean_m(R_m)                                 # average ranks across members
y_pred = argmin(R, axis=1)
# hybrid: blend rank-average with calibrated probability-average (e.g. 0.5/0.5), tune on OOF
```

**Expected benefit:** +0.01 to +0.02 over naive probability averaging when scales mismatch; essentially free to implement.

**Pitfalls / when NOT to use:** Throws away confidence magnitude — if members are well-calibrated (#2), probability averaging is strictly more informative, so rank averaging should be a fallback or a *component* of a hybrid blend, not the final answer. Ties need deterministic handling.

---

### 5. TabPFN as an additional diverse base model
**Sources:** [Hollmann et al., Nature 2025 — Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) · [TabPFN-2.5 Model Report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report) · [PriorLabs/TabPFN GitHub](https://github.com/PriorLabs/TabPFN)

**Core idea:** TabPFN is a transformer foundation model trained on ~100M synthetic datasets; for datasets ≤10k samples × 500 features it outperforms tuned XGBoost/CatBoost in a single forward pass (Nature, 1079 citations). It is a *different inductive bias* than your JEPA/CNN models (in-context learning over the whole table), it returns well-calibrated probabilities and embeddings, and it can be dropped into your OOF stack as just another member — or used as the PHE-style greedy ensemble inside itself.

**Recipe:**
```
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(device='cuda', model_string='TabPFN-v2')
for tr, va in F.split(X, y):                       # same folds as everything else
    clf.fit(X[tr], y[tr]); OOF[va, 'tabpfn'] = flatten(clf.predict_proba(X[va]))
# embeddings: extract target-column representations of the final layer as extra meta-features
```

**Expected benefit:** Strong small-data single model and a genuinely diverse member — likely +0.01 to +0.03 as part of the stack (it can match or beat your CatBoost on this data size).

**Pitfalls / when NOT to use:** Inference is slower than CatBoost (0.2s vs 0.0002s per sample); memory scales with dataset size; supports ≤10 classes natively (fine for 4). If KNN already dominates on pure data, TabPFN may not beat it — use it as a diversity source, not a replacement.

---

### 6. Logit averaging (with temperature) vs probability averaging
**Sources:** [Tassi et al. — The impact of averaging logits over probabilities on ensembles of neural networks](https://ceur-ws.org/Vol-3215/19.pdf) · [fast.ai forum: Ensembling — Logits or Probabilities?](https://forums.fast.ai/t/ensembling-logits-or-probabilities/81723) · [Logit-Based Ensemble Distribution Distillation](https://arxiv.org/abs/2305.10384)

**Core idea:** Averaging pre-softmax logits (evidence) instead of probabilities preserves accuracy but raises confidence. Empirically: for *underconfident* models it reduces ECE dramatically (12.04% → 5.40% on CIFAR10 MCD), but for *overconfident* models it inflates calibration error and can make false positives look as confident as true positives. For heterogeneous families, average logits only after per-model temperature normalization so scales are comparable.

**Recipe:**
```
T_m = per-model temperatures (technique #2)
L   = mean_m(logits_m / T_m)
P   = softmax(L)                                     # geometric-mean-like pooling
```

**Expected benefit:** Small but real gains when members are underconfident (typical for small-data DL); a cheap A/B test against probability averaging on your OOF.

**Pitfalls / when NOT to use:** Skip when models are overconfident (transformers frequently are); the fast.ai experiments show softmax-then-average is safer when score magnitudes differ wildly across base learners — which is exactly your mixed JEPA/CNN/tree case, so treat logit averaging as an experiment, not a default.

---

### 7. Entropy-weighted / confidence-weighted fusion + abstention
**Sources:** [Uncertainty-weighted semi-supervised learning with dynamic entropy (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-30069-3) · [Bayesian ensemble with entropy-weighted aggregation (Reliab. Eng. Syst. Saf. 2025)](https://www.sciencedirect.com/science/article/pii/S0951832025007756) · [Tiny Deep Ensemble (arXiv 2405.05286)](https://arxiv.org/html/2405.05286v1)

**Core idea:** Weight each member *per sample* by its confidence: `w_m = exp(-H(P_m))`, i.e., trust KNN where it is sharp and the DL models where they are sharp. For MCC specifically, an **abstention layer** is often the highest-value trick: reject the ~5–10% of samples with the lowest ensemble confidence (max probability below a threshold τ) — MCC is computed only on accepted samples, which can jump if the rejected samples are exactly the misclassified ones.

**Recipe:**
```
H_m = -sum(P_m * log(P_m), axis=1)                  # per-sample entropy per member
w_m = exp(-H_m) / sum_m exp(-H_m)                   # normalize per sample
P   = sum_m w_m[:, None] * P_m
conf = P.max(axis=1)
accepted = conf >= tau                               # tune tau on OOF for max MCC
mcc_eval = MCC(y[accepted], P[accepted].argmax(1))   # report coverage alongside
```

**Expected benefit:** +0.005 to +0.02 on MCC; especially effective because your members have sharply different confidence regimes (KNN is confident where DL is unsure and vice versa).

**Pitfalls / when NOT to use:** Entropy weighting can collapse to one member if a model is systematically overconfident — calibrate first (#2). Abstention only helps if the evaluation allows rejecting samples; if the test set must be fully labeled, use τ = 0 and keep only the weighting part.

---

### 8. Seed ensembling + snapshot/checkpoint ensembling
**Sources:** [Snapshot Ensembles: Train 1, Get M for Free (arXiv 1704.00109)](https://arxiv.org/abs/1704.00109) · [NVIDIA playbook — 100-seed XGBoost ensemble](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) · [Hyperparameter ensembles (Wenzel et al., NeurIPS 2020)](https://arxiv.org/abs/2006.13570)

**Core idea:** Retrain each DL member with 5–10 random seeds and treat each seed as an ensemble member (Lakshminarayanan-style deep ensemble); or train once with a cyclic/cosine-annealing LR and save a checkpoint at each local minimum (snapshot ensembling) — M models for the cost of one run. Kaggle evidence (Predicting Optimal Fertilizers): ensembling 100 XGBoost seeds beat any single seed steadily. This is the cheapest diversity you can add, and it matters most here because the JEPA models are under-trained — seed + snapshot diversity partly compensates.

**Recipe:**
```
for seed in [42, 123, 456, 789, 2024]:
    m = train(arch, X, y, seed=seed)                # different init + data order
    P[seed] = predict_proba(m, X_test)
P = mean(seeds)                                      # each seed = ensemble member
# Snapshot variant: cosine LR with restarts every ~40 epochs; save weights at each valley
```

**Expected benefit:** +0.005 to +0.02 for the DL members on small data; nearly free if you already have multi-seed jobs (`job_multi_seed.sh` exists in your repo).

**Pitfalls / when NOT to use:** Members from the same arch/seed-family are correlated — snapshot gains shrink for long runs with stable minima; if the base model is weak, seed ensembling averages weak biases without fixing them. Use it to *enlarge the library* for Caruana selection (#3), not as the final combiner by itself.

---

### 9. Diversity-driven ensemble selection (correlation-aware pruning, LogME, Q(D)O-ES)
**Sources:** [Efficient Diversity-Driven Ensemble for DNNs (arXiv 2112.13316)](https://arxiv.org/abs/2112.13316) · [Diversity Regularized Ensemble Pruning (DREP)](https://www.lamda.nju.edu.cn/publication/ecml12.pdf) · [Hierarchical Pruning of Deep Ensembles with Focal Diversity](https://dl.acm.org/doi/10.1145/3633286) · [LogME transferability metric](https://github.com/thuml/LogME) · [Q(D)O-ES: quality-diversity post-hoc ensemble selection](https://arxiv.org/abs/2302.02149)

**Core idea:** Your 5 JEPA-family models are likely highly correlated — an ensemble of correlated members dilutes the meta-learner. Selection should maximize *marginal gain + diversity*: pick the best single model, then repeatedly add the member with the best OOF score *and* lowest average correlation to the current selection (a λ-weighted score). LogME gives a cheap, label-dependent score of how well each model's embeddings fit the target task — useful for ranking candidates without refitting. Q(D)O-ES is the quality–diversity search AutoML uses for post-hoc ensemble selection (used inside TabPFN PHE).

**Recipe:**
```
corr = corrcoef(OOF_proba)                          # or disagreement matrix
selected = [argmax_m logloss_score(m)]
while OOF score improves:
    cand = argmax_m [ marginal_gain(m) + lam * (1 - mean(corr[m, selected])) ]
    if score(selected + cand) not better: break
# LogME ranking: score_m = logme(embeddings_m, y)   # seconds per model, no retraining
```

**Expected benefit:** Prunes redundant JEPA members → smaller, more robust stack and cleaner meta-feature space; protects against dilution when you later add 15–40 members (#3).

**Pitfalls / when NOT to use:** Diversity metrics can be gamed and, with <10 members, correlation pruning is coarse; never prune the only strong member (KNN) in favor of diversity. Q(D)O-ES / LogME assume you have embeddings — extract them from your DL models' penultimate layers.

---

### 10. Deep stacking / multi-level stacking (DSN lineage; XStacking; DSEM-NIDS)
**Sources:** [Deng & Yu — Deep Stacking Network (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/scalable-stacking-and-learning-for-building-deep-architectures/) · [Deng et al. ISCA 2012 — Parallel training of DSNs](https://www.isca-archive.org/interspeech_2012/deng12_interspeech.pdf) · [Tensor Deep Stacking Networks (TPAMI)](https://dl.acm.org/doi/10.1109/TPAMI.2012.268) · [XStacking (arXiv 2507.17650, Jul 2025)](https://arxiv.org/html/2507.17650) · [DSEM-NIDS (deep stacking ensemble)](https://www.researchgate.net/publication/392726998_DSEM-NIDS_Enhanced_Network_Intrusion_Detection_System_Using_Deep_Stacking_Ensemble_Model)

**Core idea:** You already run the modern incarnation of Deng & Yu's deep stacking (DeepStack: 18 bases → per-class Level2Nets → residual DeepMetaMLP, 0.859). The 2024–25 increments worth adding: (a) **XStacking** — enrich the meta-learner's input with each base model's per-sample SHAP explanation vector, i.e., train the level-2 model on `[OOF probabilities ⊕ SHAP features]`; validated on 29 datasets (equal/better than plain stacking on 16/17 classification sets); (b) **DSEM-NIDS**-style nested stacking (stack inside a stack) with strong per-level OOF discipline; (c) feed *classical* OOFs into your existing Level2Nets — your per-class level-2 architecture is precisely the right place for KNN's per-class confidence to be exploited.

**Recipe (XStacking increment on your DeepStack):**
```
# Level 1: OOF proba of all bases (incl. classical) — you have this
# Level 1.5: for each base m, compute SHAP values on its prediction for each sample
shap_m = explainer_m(X)                              # (n, d) attribution per base
Z = hstack([OOF, concat_m(shap_m)])                  # XStacking meta-space
L2 = Level2Net(Z, y)  -> OOF_L2                       # your existing per-class nets
L3 = LR([OOF_L2, OOF_classical, entropy_features], y)
```

**Expected benefit:** +0.005 to +0.01 over single-level stacking; SHAP features add signal exactly when base predictions are correlated (your JEPA family). Deeper stacks buy less and less per level.

**Pitfalls / when NOT to use:** Every additional level must be OOF-safe or the stack leaks and overfits the 2–10k samples; deep stacks overfit small data — keep L2/L3 heavily regularized (LR/ridge); diminishing returns after 2–3 levels; huge compute and ops cost for a fraction of a point.

---

### 11. MC dropout / deep ensemble uncertainty for probabilistic fusion
**Sources:** [Lakshminarayanan et al., NeurIPS 2017 — Deep Ensembles](https://arxiv.org/abs/1612.01474) · [Gal & Ghahramani, ICML 2016 — MC Dropout](https://arxiv.org/abs/1506.02142) · [Fort et al. — Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757) · [Tiny Deep Ensemble (arXiv 2405.05286)](https://arxiv.org/html/2405.05286v1)

**Core idea:** For each DL member, run inference T times with dropout enabled (MC dropout) or with T different seeds (deep ensemble) and use the mean as the member's probability and the variance as its uncertainty. The variance feeds entropy weighting (#7) and gives the meta-learner (#1) a per-sample uncertainty feature. Deep ensembles with random init + temperature scaling remain the 2024–25 baseline for probabilistic fusion.

**Recipe:**
```
for m in models:
    draws = [softmax(m.predict(X, dropout=True)) for _ in range(30)]
    P_m = mean(draws); V_m = var(draws)
# features: add mean(V_m) and per-class V_m to the meta-feature matrix
# or weight fusion: P = sum_m (P_m / (1 + V_m.mean(1)))  normalized
```

**Expected benefit:** Uncertainty features reliably add signal in stacking on small data; MC dropout costs no retraining (dropout must exist in the architecture — check your JEPA/CNN heads).

**Pitfalls / when NOT to use:** MC dropout underestimates uncertainty and needs correct dropout placement (before the last linear layer); it multiplies inference cost by T; if your models have no dropout (e.g., BatchNorm-heavy), prefer seed-based deep ensembles (#8) for the uncertainty estimate.

---

### 12. Pseudo-labeling with soft labels (semi-supervised boost)
**Sources:** [NVIDIA Grandmasters Playbook — pseudo-labeling section (BirdCLEF 2024 win)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

**Core idea:** Use your strongest model (KNN or the stack) to pseudo-label the unlabeled sequences with *soft* probabilities, then retrain the DL members on labeled + pseudo-labeled data. Kaggle-validated rules: the stronger the teacher the better; soft labels > hard labels; filter out low-confidence pseudo-labels; and when using k-fold, compute k sets of pseudo-labels so validation never sees labels produced by a model trained on itself. This matters doubly for you: JEPA pretraining is self-supervised and can use *every* RSSI sequence (including test-set sequences, per your roadmap Phase 4-D), and your data is 84% interfered — the unlabeled/interfered pool is exactly the pool the teacher can label.

**Recipe:**
```
teacher = fit_best(KNN or stack, X_labeled, y)           # 0.92 on pure data
for fold in F:                                            # leakage-safe
    pseudo[fold] = teacher_excluding_fold.predict_proba(X_unlabeled)
# confidence filter: keep pseudo rows where max prob > 0.8 (tune on labeled OOF)
X_aug = concat(X_labeled, X_unlabeled[kept])
y_aug = concat(y, pseudo[kept])                           # soft labels, weighted 0.5-1.0
for m in DL_models: m.fit(X_aug, y_aug)                   # also: JEPA SSL pretrain on all X
```

**Expected benefit:** +0.01 to +0.03 when an unlabeled pool exists (it does); also improves JEPA pretraining quality directly.

**Pitfalls / when NOT to use:** Leakage (handle per-fold as above); noisy pseudo-labels hurt if the teacher is weaker than the DL models (not your case — KNN is stronger); hard labels introduce confirmation bias — use soft labels and a confidence filter; if there is genuinely no unlabeled data, skip this.

---

### 13. Residual stacking (stage-2 learns stage-1 errors)
**Sources:** [NVIDIA Grandmasters Playbook — stacking section (residuals vs OOF features)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

**Core idea:** In addition to (or instead of) feeding OOF predictions as meta-features, train a stage-2 model on what stage-1 got wrong: residuals per class (one-vs-rest margins). This is the other half of the Kaggle stacking playbook and it captures systematic error patterns that feature-stacking misses. The Podcast Listening Time 1st-place solution used a 3-level stack mixing linear, GBDT, NN, and AutoML families.

**Recipe:**
```
P1 = stage1.predict_proba(X)                            # strong stack
R  = onehot(y) - P1                                     # per-class residuals
stage2 = CatBoost/ridge.fit(hstack([X_raw, P1]), R)     # predict residuals
final = P1 + alpha * stage2.predict(...)                # alpha tuned on OOF, ~0.1-0.5
```

**Expected benefit:** Small (+0.005) but reliable when stage-1 is already strong — a good final polish after #1–#5.

**Pitfalls / when NOT to use:** Residual models amplify noise on small data; alpha must be tuned or it double-counts; if stage-1 is weak, residual fitting learns noise — apply only after the stack is strong.

---

### 14. Hill climbing + retrain-on-100% (final polish)
**Sources:** [NVIDIA Grandmasters Playbook — hill climbing & extra training sections](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) · [Chris Deotte — GPU hill climbing (Calorie Expenditure 1st place)](https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930)

**Core idea:** After stacking, run a greedy weight search over the final members: start from the best single member, iteratively add/perturb weights, keep any change that improves OOF (this is Caruana selection restricted to the final shortlist — effectively what the NVIDIA team calls hill climbing). Then retrain every selected member on 100% of the data (no holdout) for the final submission — Kaggle-validated to give a last bump.

**Recipe:**
```
w = onehot(best_member)
repeat until no gain:
    for m, delta in candidates(0.05 steps):
        if OOF_score(w + delta * e_m) > OOF_score(w): w += delta * e_m; break
# final: retrain all w>0 members on 100% data, predict weighted average
```

**Expected benefit:** Final +0.003 to +0.01; the classic last-mile step that often decides 0.88 vs 0.90.

**Pitfalls / when NOT to use:** Hill climbing on a small validation set overfits — use bagged hill climbing (average over bootstrap resamples of the OOF) and stop when gains < 0.001; retrain-on-100% destroys your validation set, so do it only for the final evaluation; never hill-climb on the test set.

---

## Recommended combined pipeline (order of operations)

**Phase A — Members (parallelizable):** Fully pretrain JEPA models (roadmap Phase 1). Generate OOF + test probabilities for: 5 DL models × 5 seeds, KNN, CatBoost, GP, TabPFN (14–40 members total). Save `oof_*.npy`/`pred_*.npy` + a manifest (kaggle-tabular-forge discipline).

**Phase B — Calibrate:** Per-model temperature on a 1–5% stratified holdout; evaluate joint (ensemble-level) temperature as well. Store calibrated probabilities for all members.

**Phase C — Select:** Upgraded Caruana (CE scoring + bagged selection + top-N init) over the library; optionally prune with correlation/LogME first. Result: ~6–12 members with weights.

**Phase D — Stack:** Regularized LR (or tiny MLP) meta-learner on selected members' OOF proba + entropy + embedding features → OOF of the meta. This replaces/extends DeepStack's Level-2 with the classical-model OOFs included.

**Phase E — Blend:** Compare calibrated probability averaging vs logit averaging vs rank averaging (0.5/0.5 hybrid with rank); add entropy weighting; tune an abstention threshold for MCC if rejection is allowed.

**Phase F — Polish:** Residual stage-2 if OOF improves; pseudo-label unlabeled pool with soft labels and retrain DL members; hill-climb final weights; retrain selected members on 100% of data; final evaluation.

**Expected outcome:** 0.859 (DeepStack) + KNN/CatBoost/GP OOFs + TabPFN + calibration + selection + blend ≈ **0.88–0.91 MCC**, with the caveat that full JEPA pretraining must land first.

---

## Sources
- [Stacked Generalization: A Comprehensive Guide for 2025](https://www.shadecoder.com/topics/stacked-generalization-a-comprehensive-guide-for-2025)
- [kaggle-tabular-forge/docs/stacking.md](https://github.com/VectorPeak/kaggle-tabular-forge/blob/main/docs/stacking.md)
- [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data (NVIDIA, Sep 2025)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Grandmaster Pro Tip: Winning with Stacking using cuML (NVIDIA)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- [Caruana et al. — Ensemble Selection from Libraries of Models (ICML 2004)](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- [caruana_stacker.py — bagged Caruana selection reference implementation](https://gist.github.com/bbstats/72e3741c28c39a4fc497b04409c4fb66)
- [Greedy ensemble selection with proper scoring rules (vascadia #894)](https://github.com/petteriTeikari/vascadia/issues/894)
- [Getting the Most Out of Ensemble Selection](https://www.cs.cornell.edu/~mmunson/publications/docs/enssel_most.pdf)
- [Hollmann et al. — Accurate predictions on small data with a tabular foundation model (Nature 2025)](https://www.nature.com/articles/s41586-024-08328-6)
- [TabPFN-2.5 Model Report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- [On Joint Regularization and Calibration in Deep Ensembles (arXiv 2511.04160)](https://arxiv.org/abs/2511.04160)
- [GETS: Ensemble Temperature Scaling for Calibration (arXiv 2410.09570)](https://arxiv.org/abs/2410.09570)
- [Guo et al. — On Calibration of Modern Neural Networks (ICML 2017)](https://arxiv.org/abs/1706.04599)
- [scikit-learn Probability calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- [Tassi et al. — The impact of averaging logits over probabilities (CEUR 3215)](https://ceur-ws.org/Vol-3215/19.pdf)
- [fast.ai — Ensembling: Logits or Probabilities?](https://forums.fast.ai/t/ensembling-logits-or-probabilities/81723)
- [Logit-Based Ensemble Distribution Distillation (arXiv 2305.10384)](https://arxiv.org/abs/2305.10384)
- [Want to Win at Kaggle? Pay Attention to Your Ensembles](https://freedom251.com/want-to-win-at-kaggle-pay-attention-to-your-ensembles/)
- [Kaggle Winning Solutions](https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions)
- [Snapshot Ensembles: Train 1, Get M for Free (arXiv 1704.00109)](https://arxiv.org/abs/1704.00109)
- [Hyperparameter ensembles (Wenzel et al., arXiv 2006.13570)](https://arxiv.org/abs/2006.13570)
- [Lakshminarayanan et al. — Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (NeurIPS 2017)](https://arxiv.org/abs/1612.01474)
- [Gal & Ghahramani — Dropout as a Bayesian Approximation (ICML 2016)](https://arxiv.org/abs/1506.02142)
- [Fort et al. — Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757)
- [Tiny Deep Ensemble (arXiv 2405.05286)](https://arxiv.org/html/2405.05286v1)
- [Efficient Diversity-Driven Ensemble for DNNs (arXiv 2112.13316)](https://arxiv.org/abs/2112.13316)
- [Diversity Regularized Ensemble Pruning (DREP, ECML 2012)](https://www.lamda.nju.edu.cn/publication/ecml12.pdf)
- [Hierarchical Pruning of Deep Ensembles with Focal Diversity](https://dl.acm.org/doi/10.1145/3633286)
- [LogME — transferability assessment of embeddings](https://github.com/thuml/LogME)
- [Deng & Yu — Scalable Stacking and Learning for Building Deep Architectures (DSN)](https://www.microsoft.com/en-us/research/publication/scalable-stacking-and-learning-for-building-deep-architectures/)
- [Deng et al. — Parallel Training of Deep Stacking Networks (ISCA 2012)](https://www.isca-archive.org/interspeech_2012/deng12_interspeech.pdf)
- [Tensor Deep Stacking Networks (TPAMI 2012)](https://dl.acm.org/doi/10.1109/TPAMI.2012.268)
- [XStacking: Explanation-Guided Stacked Ensemble Learning (arXiv 2507.17650)](https://arxiv.org/html/2507.17650)
- [DSEM-NIDS: Deep Stacking Ensemble for NIDS](https://www.researchgate.net/publication/392726998_DSEM-NIDS_Enhanced_Network_Intrusion_Detection_System_Using_Deep_Stacking_Ensemble_Model)
- [Explainable deep stacking ensemble with CatBoost meta-learner (Comput. Biol. Med. 2025)](https://www.sciencedirect.com/science/article/pii/S0010482525005177)
- [Uncertainty-weighted semi-supervised learning with dynamic entropy (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-30069-3)
- [Bayesian ensemble learning via entropy-weighted aggregation (Reliab. Eng. Syst. Saf. 2025)](https://www.sciencedirect.com/science/article/pii/S0951832025007756)
- [CatBoost Embeddings features](https://catboost.ai/docs/en/features/embeddings-features)
- [Combining XGBoost and Embeddings: Hybrid Semantic Boosted Trees (MLMastery, 2025)](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)
- [Chris Deotte — GPU Hill Climbing (1st place Calorie Expenditure)](https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930)
