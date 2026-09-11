# Model cards

One card per reported model family. Following the *Model Cards for Model
Reporting* convention (Mitchell et al., 2019) as it applies to a research
artifact: what the model is, what it was trained on, how it was evaluated, and
what it must not be used for.

Parameter counts and checkpoints are verifiable with
`uv run inmotion model info <name>`; `uv run inmotion model verify` proves
loadability.

---

## World models (self-supervised)

**Members:** `t_jepa`, `ts_jepa`, `lejepa`, `cf_jepa`, `sigreg`

> The `mamba3_*` hybrids are registered and loadable but were explored rather
> than reported; no metric is claimed for them.

**Purpose.** Learn a representation of a 10-second RSSI window without labels,
then fine-tune a linear or shallow head for route classification. The premise is
that interference-invariant structure is easier to learn as a prediction task
than from 4-class supervision.

**Architecture.** A small Transformer encoder over patch/token embeddings.
T-JEPA masks tabular features; TS-JEPA masks time-series patches; CF-JEPA
predicts multiple horizons from short context; LeJEPA adds a Gaussian latent
regulariser (SIGReg). `sigreg` is the regulariser as a directly supervised CNN,
with no pretraining. Individual configurations are listed in
`docs/reference/models.md`.

**Training data.** `data/processed/dataset_augmented3.csv` (35,839 rows,
13 devices), derived from `dataset.csv`. Augmentation: class-conditioned noise
injection and intra-class mixup.

**Evaluation.** MCC on a held-out 20% stratified test split, seed 42. Stage-1
representation quality is tracked with a linear-probe MCC, not the pretraining
loss.

**Reported results.** `docs/isac/results/MASTER_RESULTS.md` §2.1–2.2.

**Limitations.**
- Trained and evaluated on 13 devices from one collection campaign, in one
  building. Device-specific RSSI offsets are a known confound: `clean_dataset.py`
  exists because absolute signal level carried more information about *which
  device* recorded a sequence than about the route.
- Leave-one-device-out performance is materially below the in-distribution
  number. The in-distribution MCC should not be read as cross-device
  generalisation.

**Out of scope.** Any safety-critical decision. This is a research classifier on
a 4-class toy mobility task, and it is explicitly subject to device and
environment shift.

---

## Supervised deep learning baselines

**Members:** `rnn`, `gru`, `lstm`, `bilstm`, `cnn`, `tcn`, `transformer`,
`mamba`, `cnn2d_rnn`, `meta_fusion`, plus HPO variants

**Purpose.** Establish what a conventional sequence model achieves on the same
input, as a reference point for the world models.

**Training data.** `data/processed/dataset.csv` (3,511 rows) for the baselines;
augmented variants for the HPO-best configurations.

**Evaluation.** Same split and metric as above.

**Limitations.**
- The HPO-best configurations were rebuilt from documented values because the
  Optuna databases were deleted (`inmotion run pipeline hpo-paper`). The
  configuration is documented; the *search trajectory* is not recoverable.
- `mamba` and the `mamba3_*` family require either the `mamba_ssm` CUDA
  extension or the pure-PyTorch fallback. The two are not numerically identical,
  and which one produced a given checkpoint is recorded in its log, not its
  weights.

**Out of scope.** As above.

---

## Ensembles

**Members:** `voting_ensemble`, `stacking_ensemble`, `deep_stack`, plus the MoE
variants

**Purpose.** Combine heterogeneous members to raise MCC. `deep_stack` is a
10-base stack with learned level-2 and meta models; `voting_ensemble` averages
member probabilities with selection weights.

**Combiner choices.** The rationale and the literature behind them are in
`../ENSEMBLE_TECHNIQUES_REPORT.md`. Briefly: Caruana-style forward selection
with replacement over the member library, temperature calibration per member on
the validation fold, and a regularised logistic-regression stacker as the
alternative.

**Evaluation.** Members are combined on out-of-fold predictions to avoid
optimistic selection; the ensemble is then scored once on the held-out test
split.

**Limitations.**
- The `deep_stack` reported artifact is `backup/DeepStackEnsemble_seed42.pt`
  (43,575,549 parameters). `MASTER_RESULTS.md` cites a path that does not exist;
  see `docs/reproducibility.md`.
- Ensemble gains are computed over members trained on the same 13-device pool and
  do not measure robustness to a new device.

**Out of scope.** As above.

---

## Classical machine learning

**Members:** LogisticRegression, RandomForest, ExtraTrees, GradientBoosting,
XGBoost, LightGBM, CatBoost, SVC, KNN, MLP, GaussianProcess, TabPFN

**Purpose.** Non-neural baselines, and the models served by the demo.

**Training data.** Same processed datasets. TabPFN is used in inference mode
over the OOF feature matrix.

**Evaluation.** Same split, same metric, plus cross-validation for stability.

**Limitations.** Standardised-RSSI models inherit the same device confound as
the DL models. GaussianProcess checkpoints are ~37 MB each and are the largest
classical artifacts.

**Out of scope.** As above.

---

## Intended use, for every card above

| | |
|---|---|
| **Intended** | research on WiFi-RSSI-based mobility classification; reproducing the reported numbers; teaching and extension |
| **Not intended** | surveillance, identification of individuals, or any operational transport decision |
| **Privacy note** | `mac` is retained to support leave-one-device-out analysis and is *not* a model feature, but MAC addresses are device identifiers. Treat the raw captures in `data/raw/` as personal data and check the dataset licence before redistribution. |
