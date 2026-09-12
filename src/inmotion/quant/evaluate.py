"""Accuracy under quantization, measured on ``data/processed/dataset.csv``.

Every number this module produces is on **dataset.csv**. It is *not* on
``dataset.icaisf.csv``, so it is **not** comparable with the MCC values in
``docs/isac/results/MASTER_RESULTS.md`` section 2.3 (lejepa 0.8850, t_jepa
0.8694, ts_jepa 0.8785, cf_jepa 0.8661, sigreg 0.8858), which were measured on
the icaisf variant. The two splits differ in size and in score; MASTER_RESULTS
is left untouched as the record of what was published on icaisf.

Otherwise the harness reuses the *same* splits and the same feature pipeline the
models were trained with:

- features from ``DLDataLoader`` with the model's own ``rich`` flag, because the
  scaler is fitted inside it and using a different fit would change the inputs;
- the 80/20 stratified split from ``inmotion.data.splits``, which is the
  convention stated in MASTER_RESULTS.

The first thing the sweep does is evaluate the unquantized model. If that does
not reproduce the recorded MCC, the harness is wrong and every later number is
meaningless, so :func:`check_baseline` fails loudly instead of publishing a
table nobody can trace.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from inmotion.data.paths import resolve as resolve_path
from inmotion.data.splits import stratified_indices
from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import DLDataLoader
from inmotion.models import registry as reg
from inmotion.quant.model import apply, apply_cast
from inmotion.quant.spec import FLOAT_PRECISIONS
from inmotion.quant.spec import resolve as resolve_spec

#: The one dataset quantization is evaluated on.
BASELINE_DATASET = "dataset.csv"

#: fp32 test MCC measured on :data:`BASELINE_DATASET` by this harness.
#: These are *measurements*, not paper numbers: MASTER_RESULTS.md section 2.3
#: reports the same models on ``dataset.icaisf.csv``, which scores differently.
#: Regenerate with ``inmotion quantize eval`` and edit only on purpose.
BASELINE_MCC: dict[str, float] = {
    "lejepa": 0.9142641066646362,
    "t_jepa": 0.8982207226632052,
    "ts_jepa": 0.9194244062588919,
    "cf_jepa": 0.9083210968710779,
    # Retrained on dataset.csv; config in
    # 20-aug/models/exotic/dataset-csv/sigreg_seed42.json. The icaisf lineage
    # checkpoint (registry `sigreg_icaisf`) scored 0.4872 here.
    "sigreg": 0.6333303415465718,
}


@dataclass
class EvalRow:
    """One (model, precision) measurement."""

    model: str
    precision: str
    bits: int
    dataset: str
    seed: int
    mcc: float
    accuracy: float
    f1_macro: float
    n_test: int
    total_mb: float
    weights_mb: float
    compression: float
    max_weight_error: float
    n_params: int
    extra: dict[str, object] = field(default_factory=dict)

    def flat(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("extra", None)  # nested dict, not a CSV column
        return row


def load_split(model_name: str, dataset: Path, seed: int):
    """Features, labels and test indices, using the training convention.

    Public because the exporter needs the same split to write golden vectors;
    reaching across modules for a private name would be a smell, not a boundary.
    """
    spec = reg.by_name(model_name)
    cfg = DLConfig()
    cfg.data_path = dataset
    cfg.rich_features = spec.rich
    if spec.rich:
        cfg.in_features = 18

    loader = DLDataLoader(cfg)
    X, y = loader.load_and_preprocess()
    _, _, test_idx = stratified_indices(y, seed)
    return X.astype(np.float32), y, test_idx, list(loader.classes_)


@torch.no_grad()
def _predict(model: torch.nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch):
        chunk = torch.from_numpy(X[start : start + batch])
        out.append(model(chunk).argmax(dim=-1).numpy())
    return np.concatenate(out)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate(
    model_name: str,
    precision: str,
    *,
    dataset: Path | None = None,
    seed: int = 42,
) -> EvalRow:
    """Measure one (model, precision) pair on the test split."""
    spec = reg.by_name(model_name)
    path = dataset or resolve_path(BASELINE_DATASET)
    X, y, test_idx, _ = load_split(model_name, path, seed)

    model = reg.load(spec, strict=True)
    if precision in FLOAT_PRECISIONS:
        quantized, report = apply_cast(model, precision)
    else:
        quantized, report = apply(model, resolve_spec(precision))

    preds = _predict(quantized, X[test_idx])
    metrics = _metrics(y[test_idx], preds)
    n_params = sum(p.numel() for p in model.parameters())

    return EvalRow(
        model=model_name,
        precision=precision,
        bits=report.bits,
        dataset=path.name,
        seed=seed,
        n_test=int(len(test_idx)),
        total_mb=round(report.total_bytes / 1e6, 6),
        weights_mb=round(report.quantized_bytes / 1e6, 6),
        compression=round(report.compression, 4),
        max_weight_error=round(report.max_error, 6),
        n_params=n_params,
        extra=report.summary(),
        **metrics,
    )


def check_baseline(rows: list[EvalRow]) -> list[str]:
    """Confirm fp32 reproduces the recorded MCC on the baseline dataset.

    The comparison is exact. The recorded values were measured by this harness
    on this split, so any movement means the model, the split or the harness
    changed and every number below is untrustworthy. Rows measured on another
    dataset cannot be compared and are skipped, but an fp32 row whose model has
    no recorded baseline is itself a problem, so a new model cannot slip in
    unchecked.

    Returns human-readable problems; empty means the harness is trustworthy.
    """
    problems: list[str] = []
    for row in rows:
        if row.precision != "fp32" or row.dataset != BASELINE_DATASET:
            continue
        expected = BASELINE_MCC.get(row.model)
        if expected is None:
            problems.append(f"{row.model}: no recorded fp32 baseline on {BASELINE_DATASET}")
            continue
        if row.mcc != expected:
            problems.append(
                f"{row.model}: fp32 MCC {row.mcc!r} vs recorded {expected!r} on {BASELINE_DATASET}"
            )
    return problems


DEFAULT_PRECISIONS: tuple[str, ...] = (
    "fp32",
    "bf16",
    "fp16",
    "int8",
    "q8",
    "q4",
    "q2",
)


def sweep(
    model_names: list[str],
    precisions: tuple[str, ...] = DEFAULT_PRECISIONS,
    *,
    dataset: Path | None = None,
    seed: int = 42,
    out_csv: Path | None = None,
) -> list[EvalRow]:
    """Evaluate every (model, precision) pair and optionally write a CSV."""
    rows: list[EvalRow] = []
    for model_name in model_names:
        for precision in precisions:
            try:
                row = evaluate(model_name, precision, dataset=dataset, seed=seed)
                rows.append(row)
                print(
                    f"  {model_name:10s} {precision:5s} mcc={row.mcc:.4f} "
                    f"acc={row.accuracy:.4f} total={row.total_mb:.3f}MB x{row.compression:.2f}"
                )
            except FileNotFoundError as exc:
                print(f"  {model_name:10s} {precision:5s} SKIP: {exc}")
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"  {model_name:10s} {precision:5s} FAIL {type(exc).__name__}: {exc}")

    if out_csv is not None and rows:
        write_csv(rows, out_csv)
    return rows


def write_csv(rows: list[EvalRow], path: Path) -> Path:
    """Write the sweep as one row per measurement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Field names come from the dataclass, so the header cannot drift from the
    # row that is written. (`extra` is excluded: it is a nested summary dict.)
    fields = [f.name for f in dataclasses.fields(EvalRow) if f.name != "extra"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.flat())
    print(f"wrote {path}")
    return path


def write_json(rows: list[EvalRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.flat() for r in rows], indent=2) + "\n")
    return path
