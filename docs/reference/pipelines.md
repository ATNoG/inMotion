# Pipeline reference

Generated from the pipelines' own `RunConfig` dataclasses, so the option
list cannot drift from what the code accepts.

Every pipeline has the same shape: a typed config and a `run(args) -> int`.
The CLI builds the config from the options below; from Python you construct
it directly:

```python
from inmotion.pipelines.exotic import RunConfig, run

run(RunConfig(model='ts_jepa', seed=42, pretrain_epochs=500))
```

Return value is the exit code, so a caller can check it.

| command | pipeline module | options |
|---|---|---:|
| `inmotion train dl` | `inmotion.pipelines.dl` | 16 |
| `inmotion train exotic` | `inmotion.pipelines.exotic` | 56 |
| `inmotion train classification` | `inmotion.pipelines.classification` | 13 |
| `inmotion train hpo-paper` | `inmotion.pipelines.hpo_paper` | 7 |
| `inmotion evaluate ensemble` | `inmotion.pipelines.ensemble` | 3 |
| `inmotion evaluate mega-ensemble` | `inmotion.pipelines.mega_ensemble` | 8 |
| `inmotion evaluate checkpoints` | `inmotion.evaluation.checkpoints` | 8 |
| `inmotion evaluate shap` | `inmotion.evaluation.shap` | 7 |
| `inmotion evaluate feature-importance` | `inmotion.evaluation.feature_importance` | 7 |
| `inmotion analyze seeds` | `inmotion.analysis.seeds` | 2 |
| `inmotion analyze interference` | `inmotion.analysis.interference` | 2 |
| `inmotion plots classical` | `inmotion.viz.regenerate` | 10 |
| `inmotion plots dl` | `inmotion.viz.regenerate_dl` | 6 |
| `inmotion data clean` | `inmotion.data.clean` | 8 |

## `inmotion train dl`

Module: `inmotion.pipelines.dl`

Typed options for the DL pipeline.

| option | type | default |
|---|---|---|
| `--seed` | `int` | `42` |
| `--epochs` | `int` | `150` |
| `--trials` | `int` | `50` |
| `--batch-size` | `int` | `64` |
| `--num-gpus` | `int` | `0` |
| `--no-wandb` | `bool` | `False` |
| `--no-optuna` | `bool` | `False` |
| `--no-moe` | `bool` | `False` |
| `--no-deepstack` | `bool` | `False` |
| `--no-meta` | `bool` | `False` |
| `--no-resume` | `bool` | `False` |
| `--wandb-project` | `str | None` | `None` |
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--models-dir` | `Path | None` | `None` |
| `--results-dir` | `Path | None` | `None` |
| `--plots-dir` | `Path | None` | `None` |


## `inmotion train exotic`

Module: `inmotion.pipelines.exotic`

Typed options for the exotic pipeline.

| option | type | default |
|---|---|---|
| `--model` | `str` | `'t_jepa'` |
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--rich-features` | `bool` | `False` |
| `--seed` | `int` | `42` |
| `--batch-size` | `int` | `128` |
| `--gpu` | `int` | `0` |
| `--pretrain-epochs` | `int` | `300` |
| `--pretrain-lr` | `float` | `0.0003` |
| `--pretrain-wd` | `float` | `0.0001` |
| `--probe-cadence` | `int` | `10` |
| `--probe-patience` | `int` | `15` |
| `--finetune-epochs` | `int` | `50` |
| `--epochs` | `int` | `150` |
| `--patience` | `int` | `25` |
| `--finetune-lr` | `float` | `0.001` |
| `--mixup-alpha` | `float` | `0.4` |
| `--label-smoothing` | `float` | `0.1` |
| `--lr` | `float` | `0.001` |
| `--finetune-only` | `bool` | `False` |
| `--checkpoint` | `Path` | `None` |
| `--unfreeze-encoder` | `bool` | `False` |
| `--d-model` | `int` | `256` |
| `--d-state` | `int` | `16` |
| `--dropout` | `float` | `0.2` |
| `--mimo-rank` | `int` | `4` |
| `--nhead` | `int` | `8` |
| `--num-layers` | `int` | `4` |
| `--dim-ff` | `int` | `512` |
| `--n-reg-tokens` | `int` | `2` |
| `--pred-dim` | `int` | `128` |
| `--pred-layers` | `int` | `2` |
| `--ema-start` | `float` | `0.99` |
| `--ema-end` | `float` | `0.996` |
| `--n-preds` | `int` | `1` |
| `--jepa-sigreg` | `float` | `0.05` |
| `--sigreg-mode` | `str` | `'sigreg'` |
| `--rdm-p` | `float` | `1.5` |
| `--pooling` | `str` | `'mean'` |
| `--cf-horizon-start` | `int` | `1` |
| `--cf-horizon-end` | `int` | `3` |
| `--cf-ctx-min` | `int` | `1` |
| `--cf-ctx-max` | `int` | `3` |
| `--num-filters` | `int` | `192` |
| `--num-blocks` | `int` | `3` |
| `--latent-dim` | `int` | `128` |
| `--sigreg-lambda` | `float` | `0.01` |
| `--hpo` | `bool` | `False` |
| `--hpo-trials` | `int` | `30` |
| `--hpo-objective` | `str` | `'mcc'` |
| `--hpo-pareto-train` | `int` | `0` |
| `--hpo-pareto-select` | `str` | `'spread'` |
| `--models-dir` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/models/exotic')` |
| `--results-dir` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/results/exotic')` |
| `--viz-dir` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/models/exotic/viz')` |
| `--no-wandb` | `bool` | `False` |
| `--wandb-project` | `str` | `'inMotion-exotic-hpo'` |


## `inmotion train classification`

Module: `inmotion.pipelines.classification`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `str` | `'data/processed/dataset.csv'` |
| `--quick` | `bool` | `False` |
| `--no-ensembles` | `bool` | `False` |
| `--optimize` | `bool` | `False` |
| `--n-trials` | `int` | `100` |
| `--cv-folds` | `int` | `5` |
| `--seed` | `int` | `42` |
| `--n-jobs` | `int` | `-1` |
| `--skip-eda` | `bool` | `False` |
| `--storage` | `str` | `'sqlite:///optuna_studies.db'` |
| `--dashboard` | `bool` | `False` |
| `--no-seed-suffix` | `bool` | `False` |
| `--suffix-after-seed` | `str` | `''` |


## `inmotion train hpo-paper`

Module: `inmotion.pipelines.hpo_paper`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--seed` | `int` | `42` |
| `--epochs` | `int` | `150` |
| `--patience` | `int` | `20` |
| `--models` | `list[str] | None` | `None` |
| `--models-dir` | `Path` | `PosixPath('models/dl')` |
| `--results-dir` | `Path` | `PosixPath('results/dl')` |


## `inmotion evaluate ensemble`

Module: `inmotion.pipelines.ensemble`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset_augmented3.csv')` |
| `--seed` | `int` | `42` |
| `--steps` | `int` | `50` |


## `inmotion evaluate mega-ensemble`

Module: `inmotion.pipelines.mega_ensemble`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--seed` | `int` | `42` |
| `--steps` | `int` | `100` |
| `--no-tabpfn` | `bool` | `False` |
| `--no-classical` | `bool` | `False` |
| `--members` | `str` | `'all'` |
| `--select` | `bool` | `False` |
| `--size-frontier` | `bool` | `False` |


## `inmotion evaluate checkpoints`

Module: `inmotion.evaluation.checkpoints`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--seed` | `int` | `42` |
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--checkpoints-dir` | `Path` | `PosixPath('models/dl')` |
| `--output` | `Path` | `None` |
| `--batch-size` | `int` | `64` |
| `--no-wandb` | `bool` | `False` |
| `--models-dir` | `Path` | `None` |
| `--optuna-db` | `str` | `'sqlite:///optuna_dl.db'` |


## `inmotion evaluate shap`

Module: `inmotion.evaluation.shap`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--model` | `str` | `'ts_jepa'` |
| `--checkpoint` | `Path` | `PosixPath('backup/ts_jepa_ft_seed42.pt')` |
| `--seed` | `int` | `42` |
| `--samples` | `int` | `50` |
| `--nsamples` | `int` | `150` |
| `--output-dir` | `Path` | `PosixPath('results')` |


## `inmotion evaluate feature-importance`

Module: `inmotion.evaluation.feature_importance`

Command-line options for the feature-importance pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path | None` | `None` |
| `--seed` | `int` | `42` |
| `--epochs` | `int` | `100` |
| `--patience` | `int` | `20` |
| `--output-dir` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/docs/icaisf/paper/images')` |
| `--checkpoints-dir` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/models/dl')` |
| `--n-repeats` | `int | None` | `None` |


## `inmotion analyze seeds`

Module: `inmotion.analysis.seeds`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--seeds` | `str` | `'3,5,42'` |
| `--output-dir` | `str` | `'seeds_results_combined'` |


## `inmotion analyze interference`

Module: `inmotion.analysis.interference`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--output-dir` | `Path` | `PosixPath('plots/interference')` |


## `inmotion plots classical`

Module: `inmotion.viz.regenerate`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--results-dir` | `str` | `None` |
| `--csv` | `str` | `None` |
| `--detailed-csv` | `str` | `None` |
| `--plots-dir` | `str` | `None` |
| `--top-n` | `int` | `15` |
| `--format` | `str` | `'pdf'` |
| `--eda` | `bool` | `False` |
| `--data` | `str` | `'data/processed/dataset.csv'` |
| `--seed` | `int` | `42` |
| `--skip-classification-plots` | `bool` | `False` |


## `inmotion plots dl`

Module: `inmotion.viz.regenerate_dl`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--results-csv` | `Path` | `None` |
| `--output-dir` | `Path` | `None` |
| `--top-n` | `int` | `15` |
| `--eda` | `bool` | `False` |
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--seed` | `int` | `42` |


## `inmotion data clean`

Module: `inmotion.data.clean`

Command-line options for this pipeline.

| option | type | default |
|---|---|---|
| `--data` | `Path` | `PosixPath('/home/andre/Desktop/UA/IT/inMotion/inMotion/data/processed/dataset.csv')` |
| `--out` | `Path` | `None` |
| `--report-dir` | `Path` | `PosixPath('results')` |
| `--within-floor` | `float` | `0.15` |
| `--lodo-floor` | `float` | `0.05` |
| `--keep-shifted` | `bool` | `False` |
| `--drop-ambiguous` | `bool` | `False` |
| `--ambiguity-thresh` | `float` | `0.85` |
