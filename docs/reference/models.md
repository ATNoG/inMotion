# Model reference

Generated from `inmotion.models.registry`. `inmotion model verify` proves
these are loadable; `inmotion model list` prints the same table.

`features` is what the model consumes at its input: `raw` is the 10 RSSI
values, `rich` is the 18-channel expansion (raw, first and second
differences, window deviation, spectral and shape summaries).

`params` is the exact module parameter count; `paper` is the figure printed
in `docs/isac/results/MASTER_RESULTS.md`. Where the two disagree the stored
checkpoint is not the model that table describes, and the gap is recorded in
`PROVENANCE_GAPS`.

Canonical model count: **18**

| name | factory | features | params | paper (M) | checkpoint |
|---|---|---|---:|---:|---|
| `cf_jepa` | `cf_jepa` | rich | 1,232,261 | 1.232 | `20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt` |
| `cf_jepa_base` | `cf_jepa` | rich | 2,153,349 | - | `20-aug/models/exotic/normal-new-ds/cf_jepa_ft_seed42.pt` |
| `deep_stack` | `deep_stack` | raw | 43,575,549 | 43.576 | `backup/DeepStackEnsemble_seed42.pt` |
| `lejepa` | `lejepa` | raw | 565,765 | 0.566 | `20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt` |
| `lejepa_base` | `lejepa` | rich | 2,215,813 | - | `20-aug/models/exotic/normal-new-ds/lejepa_ft_seed42.pt` |
| `mamba3_cnn` | `mamba3_cnn` | raw | 945,028 | - | `backup/mamba3_cnn_seed42.pt` |
| `mamba3_multiview` | `mamba3_multiview` | raw | 1,193,349 | - | `backup/mamba3_multiview_seed42.pt` |
| `mamba3_tcn` | `mamba3_tcn` | raw | 695,812 | - | `backup/mamba3_tcn_seed42.pt` |
| `mamba3_transformer` | `mamba3_transformer` | raw | 561,796 | - | `backup/mamba3_transformer_seed42.pt` |
| `sigreg` | `sigreg` | raw | 696,900 | 0.697 | `20-aug/models/exotic/dataset-csv/sigreg_seed42.pt` |
| `sigreg_after_hpo` | `sigreg` | raw | 317,460 | - | `20-aug/models/exotic/normal-new-ds/full-after-hpo/sigreg_seed42.pt` |
| `sigreg_base` | `sigreg` | raw | 474,564 | - | `20-aug/models/exotic/normal-new-ds/sigreg_seed42.pt` |
| `sigreg_icaisf` | `sigreg` | raw | 696,900 | 0.697 | `backup/sigreg_seed42.pt` |
| `sigreg_seed3` | `sigreg` | raw | 696,900 | 0.697 | `backup/sigreg_seed3.pt` |
| `sigreg_seed5` | `sigreg` | raw | 696,900 | 0.697 | `backup/sigreg_seed5.pt` |
| `t_jepa` | `t_jepa` | rich | 633,477 | 0.633 | `20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt` |
| `t_jepa_base` | `t_jepa` | rich | 2,347,653 | - | `20-aug/models/exotic/normal-new-ds/t_jepa_ft_seed42.pt` |
| `ts_jepa` | `ts_jepa` | rich | 2,153,349 | 2.153 | `20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt` |

## Architecture details

### `cf_jepa`

- factory: `cf_jepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: (factory defaults)
- note: after-HPO frontier point

### `cf_jepa_base`

- factory: `cf_jepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/cf_jepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: `dim_feedforward=512`, `num_layers=4`
- note: pre-HPO base run (2.1)

### `deep_stack`

- factory: `deep_stack`
- checkpoint: `backup/DeepStackEnsemble_seed42.pt`
- state-dict prefix: `-`
- architecture: (factory defaults)
- note: 10-base deep stack; MASTER_RESULTS cites this under the stale path models/dl/augmented/DeepStackEnsemble_seed42.pt
- **provenance gap:** MASTER_RESULTS.md §2.3 points at models/dl/augmented/DeepStackEnsemble_seed42.pt, which does not exist. The equivalent artifact used here is backup/DeepStackEnsemble_seed42.pt.

### `lejepa`

- factory: `lejepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=128`, `dim_feedforward=256`, `nhead=4`, `num_layers=4`, `pred_num_layers=3`, `sigreg_lambda=0.05`
- note: after-HPO Pareto winner

### `lejepa_base`

- factory: `lejepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/lejepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=256`, `dim_feedforward=512`, `in_channels=18`, `nhead=8`, `num_layers=4`, `pred_num_layers=2`, `sigreg_lambda=0.1`
- note: pre-HPO base run (2.1)

### `mamba3_cnn`

- factory: `mamba3_cnn`
- checkpoint: `backup/mamba3_cnn_seed42.pt`
- state-dict prefix: `-`
- architecture: (factory defaults)
- note: experimental; not in the reported results

### `mamba3_multiview`

- factory: `mamba3_multiview`
- checkpoint: `backup/mamba3_multiview_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=256`, `d_state=16`, `mimo_rank=4`, `n_mamba_layers=2`
- note: experimental; not in the reported results

### `mamba3_tcn`

- factory: `mamba3_tcn`
- checkpoint: `backup/mamba3_tcn_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=128`, `d_state=16`, `mimo_rank=4`, `n_mamba_layers=2`, `tcn_channels=128`, `tcn_depth=4`
- note: experimental; not in the reported results

### `mamba3_transformer`

- factory: `mamba3_transformer`
- checkpoint: `backup/mamba3_transformer_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=128`, `d_state=16`, `mimo_rank=4`, `nhead=4`, `num_blocks=2`
- note: experimental; not in the reported results

### `sigreg`

- factory: `sigreg`
- checkpoint: `20-aug/models/exotic/dataset-csv/sigreg_seed42.pt`
- state-dict prefix: `m.`
- architecture: `num_filters=192`
- note: retrained on dataset.csv: 192 filters / 3 blocks / latent 128, seed 42, lr 1e-3, batch 128, sigreg_lambda 0.01, 29 of 150 epochs, val MCC 0.6181, test MCC 0.6333; reproduce with `inmotion train exotic --model sigreg --data data/processed/dataset.csv --seed 42 --d-model 192 --num-layers 3 --latent-dim 128` (the arch flags are required), full config in 20-aug/models/exotic/dataset-csv/sigreg_seed42.json

### `sigreg_after_hpo`

- factory: `sigreg`
- checkpoint: `20-aug/models/exotic/normal-new-ds/full-after-hpo/sigreg_seed42.pt`
- state-dict prefix: `m.`
- architecture: `num_filters=128`
- note: 128-filter retrain at the HPO winner config (2.2)

### `sigreg_base`

- factory: `sigreg`
- checkpoint: `20-aug/models/exotic/normal-new-ds/sigreg_seed42.pt`
- state-dict prefix: `m.`
- architecture: `num_blocks=2`, `num_filters=192`
- note: pre-HPO base run (2.1), 2 residual blocks

### `sigreg_icaisf`

- factory: `sigreg`
- checkpoint: `backup/sigreg_seed42.pt`
- state-dict prefix: `m.`
- architecture: `num_filters=192`
- note: superseded by `sigreg` for dataset.csv work; this is the artifact MASTER_RESULTS.md §2.3 cites (192-filter encoder, icaisf split)

### `sigreg_seed3`

- factory: `sigreg`
- checkpoint: `backup/sigreg_seed3.pt`
- state-dict prefix: `m.`
- architecture: `num_filters=192`
- note: multi-seed diversity member

### `sigreg_seed5`

- factory: `sigreg`
- checkpoint: `backup/sigreg_seed5.pt`
- state-dict prefix: `m.`
- architecture: `num_filters=192`
- note: multi-seed diversity member

### `t_jepa`

- factory: `t_jepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=128`, `dim_feedforward=256`, `nhead=4`, `num_layers=4`, `pred_dim=64`, `pred_num_layers=3`
- note: after-HPO Pareto winner

### `t_jepa_base`

- factory: `t_jepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/t_jepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: `d_model=256`, `dim_feedforward=512`, `nhead=8`, `num_layers=4`, `pred_dim=128`, `pred_num_layers=2`
- note: pre-HPO base run (2.1); nhead/pred_dim do not affect any saved tensor shape, so they are not recoverable from the checkpoint

### `ts_jepa`

- factory: `ts_jepa`
- checkpoint: `20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt`
- state-dict prefix: `-`
- architecture: (factory defaults)
- note: base run (18ch, embed 256)

## Provenance gaps

Recorded rather than resolved. `tests/test_checkpoint_contract.py` fails if
one of these is closed by editing the paper figure or the checkpoint count
without restoring the matching artifact.

**`deep_stack`**

MASTER_RESULTS.md §2.3 points at models/dl/augmented/DeepStackEnsemble_seed42.pt, which does not exist. The equivalent artifact used here is backup/DeepStackEnsemble_seed42.pt.
