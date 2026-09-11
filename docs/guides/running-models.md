# Running a trained model

The problem this solves: previously the only way to use a saved checkpoint was
to read the training script that produced it, because the architecture and the
standardisation statistics existed nowhere else. Now a checkpoint describes
itself.

## What is available

```bash
uv run inmotion model list
```

```
name                   factory                    params  feat  present
cf_jepa                cf_jepa                 1,232,261  rich  yes
sigreg                 sigreg                    696,900   raw  yes
ts_jepa                ts_jepa                 2,153,349  rich  yes
...
```

`feat` is the input width: `raw` needs 4 channels, `rich` needs 18. Feeding the
wrong width is a bug, so the loader reads it from the model's own metadata.
`present` is whether the checkpoint is on disk — model binaries are gitignored,
so a fresh clone has none. See `docs/reports/ARTIFACTS.md`.

## Inspect a model

```bash
uv run inmotion model info ts_jepa
```

Prints the factory, exact parameter count, input width, checkpoint path, the
architecture keyword arguments, and — when the stored artifact disagrees with the
published parameter count — the provenance gap.

## Verify before trusting

```bash
uv run inmotion model verify            # every model
uv run inmotion model verify ts_jepa    # one
```

Strict-loads each checkpoint, checks the parameter count, and fails with a
non-zero exit code on any mismatch. This is the check to run before reporting
results from a checkpoint you did not train in this session.

## Predict

```bash
uv run inmotion model predict ts_jepa --input new_routes.csv
```

The input CSV needs the ten RSSI columns (`1`..`10`). Each row is one 10-second
route. Output columns: `predicted_class`, `confidence`, and `p_<label>` per class.

```bash
uv run inmotion model predict ts_jepa \
    --input new_routes.csv \
    --reference data/processed/dataset_augmented3.csv \
    --out predictions.csv
```

**Why `--reference` matters.** Standardisation statistics were not saved by the
original training code, so they are reconstructed from a reference CSV unless a
saved pipeline exists next to the weights. `--reference` should be the CSV the
model was trained on; the default is right for the shipped models, and the
command prints a warning saying which statistics were reconstructed. Predictions
computed from reconstructed statistics are not bit-identical to the original
training-time evaluation and the tool says so rather than hiding it.

## Adopt a checkpoint that has no sidecar

Checkpoints from before the sidecar convention are recovered from their weights:

```bash
uv run inmotion model adopt path/to/checkpoint.pt --name my_model
uv run inmotion model inventory 20-aug          # scan a whole directory
```

Adoption matches against the registry first, then searches a bounded,
family-specific space. It refuses to guess: TS-JEPA and CF-JEPA at the same width
have identical key sets, shapes and parameter counts, so they cannot be
distinguished from weights, and the result reports the ambiguity instead of
picking one.

Write a sidecar-tagged copy:

```bash
uv run inmotion model adopt ckpt.pt --name my_model --write artifacts/my_model
```

## Known exceptions

`*_pretrain_best.pt` are **not** classifiers. They hold a `predictor` plus an
`encoder`, so there is no single architecture to adopt. Resume them through the
pipeline:

```bash
inmotion run pipeline exotic -- --model ts_jepa \
    --checkpoint 20-aug/models/exotic/normal-new-ds/ts_jepa_pretrain_best.pt
```

`inmotion model inventory` reports these as pre-training bundles with their part
counts rather than failing.

## Using a model from Python

```python
from inmotion.models import registry

spec = registry.by_name("sigreg")
model = registry.load(spec, strict=True)      # raised on any mismatch

from inmotion.data.preprocessing import FeaturePipeline
import pandas as pd, torch

pipeline = FeaturePipeline.fit("data/processed/dataset.csv", rich=spec.rich)
X = pipeline.transform(pd.read_csv("new_routes.csv"))
with torch.no_grad():
    probabilities = torch.softmax(model(torch.from_numpy(X)), dim=-1)
```

## The demo

`demo/` serves live predictions from a phone walking past an access point:

```bash
uv run python demo/main.py       # http://127.0.0.1:8000/teacher
```

It uses the joblib models in `models/` and its own lightweight dependency set.
See `demo/README.md`.
