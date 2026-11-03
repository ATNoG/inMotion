"""Canonical edge accuracy, scored from the artifacts themselves.

The campaign's accuracy numbers must describe what a router actually loads, so
every deployable row is measured by reading ``model.safetensors`` back, undoing
the exporter's transforms and running the model on the 703-row test split — not
by trusting the in-memory quantizer that wrote the artifact. The in-memory model
is still built, but only as a *witness*: if the artifact and the quantizer
disagree on more than one prediction the run stops, because at that point one of
the two is wrong and neither number can be published.

Uncertainty comes from 2 000 bootstrap resamples of the test rows (seed 42),
computed once per model and reused for every precision, so the paired ΔMCC is a
difference of the *same* resamples rather than two independent intervals that
happen to overlap.

The cast precisions (bf16/fp16/fp8) have no artifact representation — the
runtime reads F32/I32/U8 — so they appear only as an appended accuracy section
with ``runtime_loadable=false`` and no byte count.

Nothing here is device-specific: the device workstream replays these same
windows and compares its output against ``labels.txt``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from inmotion.data.paths import resolve as resolve_path
from inmotion.edge import contracts as c
from inmotion.models import registry as reg
from inmotion.models.adopt import resolve_model
from inmotion.quant.evaluate import load_split
from inmotion.quant.export import (
    META_NAME,
    WEIGHTS_NAME,
    dequantize_artifact,
    export,
    rebuild_state_dict,
    write_golden,
)
from inmotion.quant.model import CAST_BYTES, apply, apply_cast
from inmotion.quant.producers import parse_in_house_scheme

#: Bootstrap protocol. Fixed by the campaign contract, not a tuning knob.
N_RESAMPLES = 2000
CI_PERCENTILES = (2.5, 97.5)

#: Classes in the dataset split; the confusion tables are square in this.
N_CLASSES = 4


#: One artifact directory per (model, precision).
def _artifact_dir(artifacts_dir: Path, model: str, precision: str) -> Path:
    return Path(artifacts_dir) / f"{model}-{precision}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> Path:
    """Write ``rows`` under exactly ``columns``, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


# ── Artifact export ───────────────────────────────────────────────────────


def export_artifacts(
    artifacts_dir: Path,
    models: tuple[str, ...] = c.MODELS,
    precisions: tuple[str, ...] = c.PRECISIONS_DEPLOYABLE,
) -> Path:
    """Write every ``<model>-<precision>/`` artifact plus ``artifacts.csv``.

    The golden vectors are written with the same tolerance the Rust parity test
    uses (2e-4), from the artifact's own quantized weights. Cast precisions are
    not artifact formats; they are refused here with the exporter's own error
    rather than recorded as a failed cell, because a caller asking for one has
    confused the supplementary accuracy table with the runtime format.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for model in models:
        for precision in precisions:
            if precision in c.PRECISIONS_CAST:
                # `parse_in_house_scheme` is the exporter's own refusal, raised
                # before a directory is created; let it propagate uncaught.
                parse_in_house_scheme(precision)
                raise ValueError(f"{precision} is cast-only and cannot be written as an artifact")

            out = _artifact_dir(artifacts_dir, model, precision)
            quant = None if precision == "fp32" else precision
            try:
                report = export(model, out, quant=quant)
                write_golden(model, out, quant=quant)
            except Exception as exc:  # noqa: BLE001 - recorded, then evaluated loudly later
                rows.append(
                    {
                        "model": model,
                        "precision": precision,
                        "runtime_loadable": "true",
                        "path": str(out),
                        "bytes": -1,
                        "sha256": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300].replace("\n", " "),
                    }
                )
                continue

            rows.append(
                {
                    "model": model,
                    "precision": precision,
                    "runtime_loadable": "true",
                    "path": str(out),
                    "bytes": int(report.total_bytes),
                    "sha256": _sha256_file(out / c.ARTIFACT_HASH_FILE),
                    "status": "ok",
                    "error": "",
                }
            )

    _write_csv(artifacts_dir / "artifacts.csv", c.COLUMNS["artifacts"], rows)
    return artifacts_dir


# ── Metrics and uncertainty ───────────────────────────────────────────────


def _predict(model: torch.nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            chunk = torch.from_numpy(X[start : start + batch])
            out.append(model(chunk).argmax(dim=-1).numpy())
    return np.concatenate(out)


def _resample_indices(n: int, seed: int = c.SEED) -> np.ndarray:
    """The resampled test-row indices, one seed per model.

    Re-seeded per model so every model is compared against the *same* resamples;
    within a model the matrix is reused for fp32 and every precision, which is
    what makes ΔMCC a paired statistic.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(N_RESAMPLES, n))


def _confusions(codes: np.ndarray, resamples: np.ndarray) -> np.ndarray:
    """(R, 4, 4) confusion counts, one matrix per bootstrap resample."""
    out = np.empty((resamples.shape[0], N_CLASSES, N_CLASSES), dtype=np.float64)
    for r, idx in enumerate(resamples):
        out[r] = np.bincount(codes[idx], minlength=N_CLASSES * N_CLASSES).reshape(
            N_CLASSES, N_CLASSES
        )
    return out


def _vector_metrics(cms: np.ndarray) -> dict[str, np.ndarray]:
    """MCC / accuracy / macro-F1 for a stack of confusion matrices.

    Computed from the confusion counts so 2 000 resamples cost one bincount each
    instead of a sklearn call per metric. Point estimates never use this: those
    come from the sklearn functions the recorded baseline was measured with, so
    fp32 is comparable bit-for-bit.
    """
    total = cms.sum(axis=(1, 2))
    true_counts = cms.sum(axis=2)
    pred_counts = cms.sum(axis=1)
    tp = cms.diagonal(axis1=1, axis2=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_f1 = np.where(
            (true_counts + pred_counts) > 0,
            2.0 * tp / (true_counts + pred_counts),
            0.0,
        )
        numerator = tp.sum(axis=1) * total - (pred_counts * true_counts).sum(axis=1)
        denominator = np.sqrt(
            (total**2 - (pred_counts**2).sum(axis=1)) * (total**2 - (true_counts**2).sum(axis=1))
        )
        mcc = np.where(denominator > 0, numerator / denominator, 0.0)

    return {
        "mcc": mcc,
        "accuracy": np.where(total > 0, tp.sum(axis=1) / total, 0.0),
        "f1_macro": per_class_f1.mean(axis=1),
    }


def _interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, CI_PERCENTILES)
    return float(lo), float(hi)


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


# ── Artifact scoring ──────────────────────────────────────────────────────


def _artifact_weights(artifact_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """The artifact's metadata and its dequantized float tensors.

    All the storage transforms (bit-unpacking, the folded bias, per-group
    metadata expansion) live in
    :func:`inmotion.quant.export.dequantize_artifact`, which is the artifact's
    inverse and is shared with every other consumer. This function only loads
    the two files.
    """
    meta = json.loads((artifact_dir / META_NAME).read_text())
    arrays = load_file(str(artifact_dir / WEIGHTS_NAME))
    return meta, dequantize_artifact(meta, arrays)


def _load_artifact_model(
    spec, artifact_dir: Path
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], int]:
    """A fresh model carrying the artifact's weights, plus its payload size."""
    meta, weights = _artifact_weights(artifact_dir)
    model = reg.load(spec, strict=True).eval()
    state = rebuild_state_dict(model, weights, spec.prefix)
    model.load_state_dict(state, strict=True)
    model.eval()
    weights_path = artifact_dir / WEIGHTS_NAME
    payload = weights_path.stat().st_size + (artifact_dir / META_NAME).stat().st_size
    return model, state, int(payload)


def _quantized_in_memory(
    fp32_model: torch.nn.Module, precision: str
) -> tuple[torch.nn.Module, float]:
    """The witness model for one precision, plus its max weight error."""
    if precision == "fp32":
        return fp32_model, 0.0
    if precision in CAST_BYTES:
        model, report = apply_cast(fp32_model, precision)
        return model, float(report.max_error)
    model, report = apply(fp32_model, parse_in_house_scheme(precision))
    return model, float(report.max_error)


def _max_weight_error(state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    """Largest per-tensor deviation between the artifact and the fp32 model."""
    errors: list[float] = []
    for name, tensor in state.items():
        ref = reference.get(name)
        if ref is None or ref.shape != tensor.shape:
            continue
        errors.append(float((ref.float() - tensor.float()).abs().max()))
    return max(errors, default=0.0)


def _metrics_section(
    *,
    model: str,
    checkpoint_sha256: str,
    precision: str,
    runtime_loadable: str,
    n_params: int,
    artifact_bytes: int,
    effective_bits: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp32_pred: np.ndarray,
    fp32_boot: dict[str, np.ndarray],
    resamples: np.ndarray,
    max_weight_error: float,
    artifact_sha256: str,
) -> dict[str, object]:
    """One accuracy row plus the CIs it shares with the fp32 baseline."""
    point = _point_metrics(y_true, y_pred)
    boot = _vector_metrics(_confusions(y_true * N_CLASSES + y_pred, resamples))
    mcc_lo, mcc_hi = _interval(boot["mcc"])
    acc_lo, acc_hi = _interval(boot["accuracy"])
    f1_lo, f1_hi = _interval(boot["f1_macro"])

    delta_boot = boot["mcc"] - fp32_boot["mcc"]
    delta_lo, delta_hi = _interval(delta_boot)
    return {
        "model": model,
        "checkpoint_sha256": checkpoint_sha256,
        "precision": precision,
        "runtime_loadable": runtime_loadable,
        "n_params": n_params,
        "artifact_bytes": artifact_bytes,
        "effective_bits": effective_bits,
        "mcc": point["mcc"],
        "mcc_lo": mcc_lo,
        "mcc_hi": mcc_hi,
        "accuracy": point["accuracy"],
        "accuracy_lo": acc_lo,
        "accuracy_hi": acc_hi,
        "f1_macro": point["f1_macro"],
        "f1_lo": f1_lo,
        "f1_hi": f1_hi,
        "delta_mcc": point["mcc"] - float(matthews_corrcoef(y_true, fp32_pred)),
        "delta_lo": delta_lo,
        "delta_hi": delta_hi,
        "max_weight_error": max_weight_error,
        "n_test": int(len(y_true)),
        "dataset": c.DATASET,
        "seed": c.SEED,
        "artifact_sha256": artifact_sha256,
    }


def _per_class_rows(
    model: str, precision: str, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> list[dict[str, object]]:
    labels = list(range(len(classes)))
    precision_c, recall_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows: list[dict[str, object]] = []
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        rows.append(
            {
                "model": model,
                "precision": precision,
                "class": name,
                "support": int(support[i]),
                "precision_c": float(precision_c[i]),
                "recall_c": float(recall_c[i]),
                "f1_c": float(f1_c[i]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def _confusion_rows(
    model: str, precision: str, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> list[dict[str, object]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return [
        {
            "model": model,
            "precision": precision,
            "true_class": truth,
            "predicted_class": predicted,
            "count": int(cm[i, j]),
        }
        for i, truth in enumerate(classes)
        for j, predicted in enumerate(classes)
    ]


def evaluate_artifacts(
    artifacts_dir: Path,
    tables_dir: Path,
    models: tuple[str, ...] = c.MODELS,
    precisions: tuple[str, ...] = c.PRECISIONS_DEPLOYABLE,
) -> dict[str, Path]:
    """Score every artifact and write ``accuracy.csv``, ``per_class.csv``, ``confusion.csv``.

    Deployable rows are measured from the artifact's own weights; the cast-only
    rows are appended from an in-memory cast because no artifact form exists.
    Raises if any artifact and its in-memory witness disagree on more than one
    of the 703 predictions, or if fp32 does not reproduce ``BASELINE_MCC`` —
    both mean the tables below would be untraceable.
    """
    artifacts_dir = Path(artifacts_dir)
    tables_dir = Path(tables_dir)
    dataset = resolve_path(c.DATASET)

    accuracy_rows: list[dict[str, object]] = []
    cast_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    problems: list[str] = []

    for model in models:
        spec = resolve_model(model)
        checkpoint_sha256 = _sha256_file(spec.abs_path)
        X, y, test_idx, classes = load_split(model, dataset, c.SEED)
        if len(test_idx) != c.N_TEST:
            raise RuntimeError(
                f"{model}: test split has {len(test_idx)} rows, contract expects {c.N_TEST}"
            )
        x_test = X[test_idx]
        y_true = y[test_idx]

        fp32_model = reg.load(spec, strict=True).eval()
        fp32_pred = _predict(fp32_model, x_test)
        fp32_mcc = float(matthews_corrcoef(y_true, fp32_pred))
        expected = c.BASELINE_MCC[model]
        if fp32_mcc != expected:
            problems.append(f"{model}: fp32 MCC {fp32_mcc!r} != BASELINE_MCC {expected!r}")
        n_params = int(sum(p.numel() for p in fp32_model.parameters()))
        resamples = _resample_indices(len(test_idx))
        fp32_boot = _vector_metrics(_confusions(y_true * N_CLASSES + fp32_pred, resamples))
        reference = dict(fp32_model.state_dict())

        for precision in precisions:
            artifact_dir = _artifact_dir(artifacts_dir, model, precision)
            if not (artifact_dir / WEIGHTS_NAME).is_file():
                raise FileNotFoundError(
                    f"missing artifact {artifact_dir}: call accuracy.export_artifacts() "
                    f"(or `uv run inmotion edge artifacts`) first"
                )
            artifact_model, state, artifact_bytes = _load_artifact_model(spec, artifact_dir)
            artifact_pred = _predict(artifact_model, x_test)

            witness, _witness_error = _quantized_in_memory(fp32_model, precision)
            witness_pred = _predict(witness, x_test)
            mismatches = int((artifact_pred != witness_pred).sum())
            if mismatches > 1:
                raise RuntimeError(
                    f"{model}-{precision}: artifact and in-memory quantization disagree on "
                    f"{mismatches}/{len(test_idx)} predictions; the artifact is not the "
                    f"model this table would describe"
                )

            if precision == "fp32" and float(matthews_corrcoef(y_true, artifact_pred)) != expected:
                problems.append(
                    f"{model}: fp32 artifact MCC "
                    f"{float(matthews_corrcoef(y_true, artifact_pred))!r} != BASELINE_MCC "
                    f"{expected!r}"
                )

            accuracy_rows.append(
                _metrics_section(
                    model=model,
                    checkpoint_sha256=checkpoint_sha256,
                    precision=precision,
                    runtime_loadable="true",
                    n_params=n_params,
                    artifact_bytes=artifact_bytes,
                    effective_bits=c.effective_bits(artifact_bytes, n_params),
                    y_true=y_true,
                    y_pred=artifact_pred,
                    fp32_pred=fp32_pred,
                    fp32_boot=fp32_boot,
                    resamples=resamples,
                    max_weight_error=_max_weight_error(state, reference),
                    artifact_sha256=_sha256_file(artifact_dir / c.ARTIFACT_HASH_FILE),
                )
            )
            per_class_rows.extend(_per_class_rows(model, precision, y_true, artifact_pred, classes))
            confusion_rows.extend(_confusion_rows(model, precision, y_true, artifact_pred, classes))

        # The cast section is appended after every deployable row. These have no
        # artifact, so byte columns say "not measured" (-1), never zero.
        for precision in c.PRECISIONS_CAST:
            witness, error = _quantized_in_memory(fp32_model, precision)
            pred = _predict(witness, x_test)
            accuracy_rows.append(
                _metrics_section(
                    model=model,
                    checkpoint_sha256=checkpoint_sha256,
                    precision=precision,
                    runtime_loadable="false",
                    n_params=n_params,
                    artifact_bytes=-1,
                    effective_bits=float(CAST_BYTES[precision] * 8),
                    y_true=y_true,
                    y_pred=pred,
                    fp32_pred=fp32_pred,
                    fp32_boot=fp32_boot,
                    resamples=resamples,
                    max_weight_error=error,
                    artifact_sha256="",
                )
            )

    if problems:
        raise RuntimeError("accuracy baseline check failed: " + "; ".join(problems))

    paths = {
        "accuracy": _write_csv(
            tables_dir / "accuracy.csv", c.COLUMNS["accuracy"], accuracy_rows + cast_rows
        ),
        "per_class": _write_csv(
            tables_dir / "per_class.csv", c.COLUMNS["per_class"], per_class_rows
        ),
        "confusion": _write_csv(
            tables_dir / "confusion.csv", c.COLUMNS["confusion"], confusion_rows
        ),
    }
    return paths


# ── Windows and labels for the device replay ─────────────────────────────


def _write_text(path: Path, text: str) -> Path:
    """Write ``text`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def export_windows(windows_dir: Path, models: tuple[str, ...] = c.MODELS) -> dict[str, Path]:
    """Write the replayed test windows, their labels and the class names.

    One directory per model: ``windows.txt`` has one line per test sample holding
    every flattened ``seq_len * channels`` value, ``labels.txt`` the class
    indices, and ``classes.json`` the names in index order. The device replay
    reads exactly these files, so the split here is the split the accuracy table
    was scored on.
    """
    windows_dir = Path(windows_dir)
    windows_dir.mkdir(parents=True, exist_ok=True)
    dataset = resolve_path(c.DATASET)
    paths: dict[str, Path] = {}

    for model in models:
        X, y, test_idx, classes = load_split(model, dataset, c.SEED)
        if len(test_idx) != c.N_TEST:
            raise RuntimeError(
                f"{model}: test split has {len(test_idx)} rows, contract expects {c.N_TEST}"
            )
        windows = X[test_idx].reshape(len(test_idx), -1)

        out = windows_dir / model
        out.mkdir(parents=True, exist_ok=True)
        body = "\n".join(" ".join(f"{value:.9g}" for value in row) for row in windows)
        paths[f"{model}/windows"] = _write_text(out / "windows.txt", body + "\n")
        labels = "\n".join(str(int(label)) for label in y[test_idx])
        paths[f"{model}/labels"] = _write_text(out / "labels.txt", labels + "\n")
        classes_json = json.dumps(list(classes), indent=2) + "\n"
        paths[f"{model}/classes"] = _write_text(out / "classes.json", classes_json)

    return paths
