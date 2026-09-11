# Architecture

## What the system does

A fixed WiFi access point records the signal strength (RSSI) of nearby devices
once per second. A 10-second window of those readings is classified into one of
four route labels:

| label | meaning |
|---|---|
| `AA` | stayed inside the bus |
| `BB` | stayed at the stop |
| `AB` | alighted (inside → outside) |
| `BA` | boarded (outside → inside) |

The task is sequence classification on 10 numbers. Everything else in the
repository exists to collect those numbers honestly, train models on them,
combine those models, and report the result reproducibly.

The `mamba3_*` hybrids are registered and loadable but were explored rather
than reported, so no metric is claimed for them.

## Data flow

```
field collection
      │  Wavecom .txt dumps (one JSON-ish dict per second)
      ▼
data/raw/                        inmotion data export
      │                                │  group MACs, pivot 10 readings to columns
      ▼                                ▼
data/interim/  ──────────────►  data/processed/
  one CSV per session             merged, labelled datasets
                                       │
                       inmotion data merge / clean / augment
                                       │
                                       ▼
                                 training pipelines
```

`inmotion.data.paths` owns this layout. `resolve()` searches the tiers, so a bare
filename such as `dataset.csv` still works from any working directory.

## Feature construction

A model never sees the 10 raw RSSI values directly. `inmotion.data.preprocessing`
standardises them and expands them into channels:

| channels | name | contents |
|---|---|---|
| 4 | `raw` | standardised RSSI, Δ, Δ², window deviation |
| 18 | `rich` | the 4 above plus spectral, statistical and shape summaries |

The channel count is a property of the model, recorded in its registry entry and
sidecar as `rich: bool`. Inference reads it so a model is never fed the wrong
width.

**The standardisation statistics are part of the artifact.** They were not
saved by the original training code, and two incompatible conventions existed
(`src/inmotion/dl/data_loader.py` fitted on the whole CSV before splitting, `mega_ensemble`
fitted on the training fold only), which made a saved model impossible to re-run
without its exact training CSV. `FeaturePipeline` now persists them, and
reconstructed statistics are labelled as reconstructed.

## Model families

| family | models | trained by |
|---|---|---|
| classical ML | LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, SVC, KNN, MLP, GaussianProcess | `inmotion train classification` |
| supervised DL | RNN, GRU, LSTM, BiLSTM, CNN, TCN, Transformer, Mamba, CNN2DRNN, MetaFusion, Autoformer | `inmotion train dl` |
| ensembles | voting, stacking, DeepStackEnsemble, MoE | `inmotion train dl` |
| world models (SSL) | T-JEPA, TS-JEPA, LeJEPA, CF-JEPA, SIGReg | `inmotion train exotic` |
| foundation model | TabPFN (inference only) | `inmotion evaluate mega-ensemble` |

## The registry

`inmotion.models.registry` is the single source of truth for what can be loaded.
An entry records the factory, its keyword arguments, the state-dict prefix, the
expected parameter count, and the feature width:

```python
ModelSpec("ts_jepa", "ts_jepa", {}, .../ts_jepa_ft_seed42.pt, 2_153_349, rich=True)
```

Before this, each model was rebuilt by a `_build_*` function — 27 of them across
three files — whose constants had been recovered by comparing tensor shapes.
They disagreed with each other, and loading used `strict=False`, so a mismatch
produced a model with randomly initialised layers that still returned numbers.

Loading is now `strict=True` and the parameter count is checked. A forward pass
is fingerprinted per model, so a change that still loads but alters the output is
caught too.

## Artifacts

A checkpoint alone is not usable. `inmotion.models.checkpoint` writes a sidecar
next to the weights recording everything needed to rebuild and interpret them:

```
artifacts/<name>/
    model.pt            weights
    model.json          Sidecar: factory, arch, params, prefix, classes, provenance
    preprocessing.json  FeaturePipeline metadata
    preprocessing.npz   standardisation statistics
```

`load()` rebuilds, loads strictly, and verifies the parameter count against the
sidecar. If the two disagree it raises rather than returning a partly-random
model.

## CLI and pipelines are separate layers

The interface and the computation are deliberately split.

```
src/inmotion/cli/          the interface
    model.py   datasets.py   data.py    train.py
    evaluate.py   analyze.py   plots.py   doctor.py
                     |
                     |  builds a RunConfig
                     v
src/inmotion/pipelines/    the computation
    <pipeline>/config.py     @dataclass RunConfig
    <pipeline>/pipeline.py   run(args) -> int
```

Every pipeline exposes exactly two things:

```python
RunConfig        # a dataclass of typed options
run(args) -> int # the work, returning an exit code
```

Why this matters:

- **No argument parsing in the pipelines.** They cannot see `sys.argv`, so they
  can be called from a notebook, a sweep script or a test as easily as from the
  shell.
- **The CLI holds no logic.** Re-arranging commands never touches a pipeline.
- **The option surface is introspectable.** `docs/reference/pipelines.md` is
  generated by reading the dataclasses, so the documented defaults are the real
  ones.

Driving a pipeline from Python:

```python
from inmotion.pipelines.exotic import RunConfig, run

run(RunConfig(model="ts_jepa", seed=42, pretrain_epochs=500))
```

Two pipelines are too large for one file and are packages, split by
responsibility:

| package | modules |
|---|---|
| `pipelines/dl` | `config`, `devices`, `builders`, `workers`, `pipeline` |
| `pipelines/exotic` | `config`, `devices`, `data`, `builders`, `training`, `hpo`, `hpo_jepa`, `hpo_sigreg`, `hpo_mamba3`, `pipeline` |

`workers` holds the spawn-safe functions that `ProcessPoolExecutor` needs at
module level, which is why they are not methods on a class. `python -m
inmotion.pipelines.<name>` runs the same CLI through a `__main__.py`.

## What is deliberately not shared

`demo/app/wifi_scan.py` duplicates a scanner that also exists in the archive.
This is intentional: the demo image is fastapi + scikit-learn, and depending on
the root package would pull in the torch/cu126 stack — several gigabytes — to
read a socket. The demo is the maintained copy.
