"""Quantization statistics tables, one row per observation, no hand-entered number.

The figures read only the CSVs this module writes, so a plotted value can always
be traced back to a measurement and to the command that produced it. That is the
whole point of the module: an accuracy curve that nobody can regenerate is not
evidence.

Two artifacts feed the tables:

``results/quant/sweep_dataset.csv``
    Accuracy and size per (model, precision), written by ``inmotion quantize
    eval``. The fp32 rows are re-checked against
    :data:`inmotion.quant.evaluate.BASELINE_MCC` here as well, using that
    module's own :func:`~inmotion.quant.evaluate.check_baseline`, so the tables
    cannot silently disagree with the baseline t1 enforces.

``results/quant/quant_reports.json``
    The raw per-tensor quantization reports captured by :func:`collect` from the
    same :func:`inmotion.quant.model.apply` call the sweep measures. This is what
    makes the coverage table truthful: ``apply`` records every weight it
    quantized *and* every parameter it refused, with the reason, so the tables
    can state what fraction of parameters actually got quantized instead of
    assuming all of them did. (That assumption is what hid
    ``torch.ao.quantization`` leaving ``sigreg`` at 0.97x: a model whose tensors
    are skipped reports a compression ratio it never earned.)

Reading the coverage table
--------------------------
``scheme_kind`` says which rule applied. ``affine`` rows quantize only the weight
*matrices* (2-D and up), so ``frac_params_quantized`` is a real fraction of the
parameters and ``scope=tensor`` rows name every parameter that was left alone
with the reason. ``cast`` rows (fp32/fp16/bf16) round *every* parameter to the
target width, so nothing is skipped: ``frac_params_quantized`` is 1.0 there
because all parameters are stored at the row's precision, and ``records`` still
lists the matrices so the per-layer table has the same rows it has for affine.

Latency
-------
:func:`latency_rows` ingests *measurement artifacts* — JSON files dropped into
``results/quant/`` by a benchmark run, e.g. the committed
``edge_router_j3455.json``. Nothing is transcribed by hand; a device appears in
the table exactly when a run wrote about it. When ``--latency-iters`` is set, a
torch reference measurement on this machine is added under ``runtime=torch-cpu``
so the dev-machine column is not empty before the Rust kernels land; the router
rows name ``runtime=x86_64-unknown-linux-musl``. The reference rows are fp32 only:
on the dequantizing torch path the precision does not change what is executed.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import platform
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from inmotion.data.paths import resolve as resolve_path
from inmotion.models import registry as reg
from inmotion.quant.evaluate import (
    BASELINE_DATASET,
    DEFAULT_PRECISIONS,
    EvalRow,
    check_baseline,
    load_split,
)
from inmotion.quant.model import apply, apply_cast
from inmotion.quant.spec import FLOAT_PRECISIONS
from inmotion.quant.spec import resolve as resolve_spec

#: Default artifact locations, relative to the repository root.
SWEEP_CSV = Path("results/quant/sweep_dataset.csv")
RAW_JSON = Path("results/quant/quant_reports.json")
MANIFEST_JSON = Path("results/quant/stats_manifest.json")

#: The five tables, in the order the report presents them.
TABLE_FILES: dict[str, str] = {
    "size": "stats_size.csv",
    "accuracy": "stats_accuracy.csv",
    "coverage": "stats_coverage.csv",
    "latency": "stats_latency.csv",
    "layers": "stats_layers.csv",
}

#: JSON files in ``results/quant/`` that are inputs or provenance, not device
#: benchmarks. Anything else matching the device pattern is ingested.
_LATENCY_SKIP = {RAW_JSON.name, MANIFEST_JSON.name, "sweep_dataset.json", "golden.json"}

#: A Rust bench feeds a deterministic synthetic window, not a dataset row; the
#: dataset/seed columns identify the study the row belongs to, and this says so.
_SYNTHETIC_NOTE = "synthetic bench input; dataset/seed name the sweep context"

_INT_COLUMNS = {"bits", "seed", "n_test", "n_params"}
_STR_COLUMNS = {"model", "precision", "dataset"}


# ---------------------------------------------------------------------------
# Raw capture
# ---------------------------------------------------------------------------
def _module_classes(model: torch.nn.Module) -> dict[str, str]:
    """Map every parameter name to ``<module type>.<param name>``.

    The class, not the name suffix, is what the coverage table reports: the
    interesting skip is ``MultiheadAttention.in_proj_weight`` (a dense 768x256
    matrix), and a suffix-based label would call it a plain weight.
    """
    classes: dict[str, str] = {}
    for module_name, module in model.named_modules():
        cls = type(module).__name__
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{param_name}" if module_name else param_name
            classes[full] = f"{cls}.{param_name}"
    return classes


def _role(name: str) -> str:
    if name.endswith(".weight"):
        return "weight"
    if name.endswith(".bias"):
        return "bias"
    return "param"


def collect(
    models: list[str],
    precisions: tuple[str, ...] = DEFAULT_PRECISIONS,
    *,
    dataset: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantize each (model, precision) and capture the full per-tensor report.

    Returns the payload that is written to :data:`RAW_JSON`. No forward pass and
    no accuracy measurement happens here — that is the sweep's job, and
    duplicating it would only create a second, drifting source of truth.
    """
    path = dataset or resolve_path(BASELINE_DATASET)
    entries: list[dict[str, Any]] = []
    for model_name in models:
        base = reg.load(reg.by_name(model_name), strict=True)
        classes = _module_classes(base)
        n_params_total = sum(int(p.numel()) for p in base.parameters())

        for precision in precisions:
            if precision in FLOAT_PRECISIONS:
                _, report = apply_cast(base, precision)
            else:
                _, report = apply(base, resolve_spec(precision))

            records = []
            for record in report.records:
                row = dataclasses.asdict(record)
                row["shape"] = list(row["shape"])
                row["tensor_class"] = classes.get(record.name, type(base).__name__)
                row["role"] = _role(record.name)
                records.append(row)

            skipped = [
                {
                    "name": name,
                    "reason": reason,
                    "tensor_class": classes.get(name, "unknown"),
                    "role": _role(name),
                    "n_values": int(dict(base.named_parameters())[name].numel()),
                }
                for name, reason in report.skipped.items()
            ]

            entries.append(
                {
                    "model": model_name,
                    "precision": precision,
                    "kind": "cast" if precision in FLOAT_PRECISIONS else "affine",
                    "bits": report.bits,
                    "spec": report.spec_label,
                    "dataset": path.name,
                    "seed": seed,
                    "n_params_total": n_params_total,
                    "records": records,
                    "skipped": skipped,
                    "fp32_weight_bytes": report.fp32_weight_bytes,
                    "unquantized_weight_bytes": report.unquantized_weight_bytes,
                    "quantized_bytes": report.quantized_bytes,
                    "packed_bytes": report.packed_bytes,
                    "skipped_param_bytes": report.skipped_param_bytes,
                    "buffer_bytes": report.buffer_bytes,
                    "total_bytes": report.total_bytes,
                    "compression": report.compression,
                    "max_error": report.max_error,
                }
            )
            print(
                f"  {model_name:10s} {precision:5s} "
                f"tensors={len(records):2d} skipped={len(skipped):2d} "
                f"total={report.total_bytes / 1e6:.3f}MB x{report.compression:.2f}"
            )

    return {
        "command": recorded_command(),
        "hostname": socket.gethostname(),
        "dataset": path.name,
        "seed": seed,
        "models": models,
        "precisions": list(precisions),
        "reports": entries,
    }


def recorded_command() -> str:
    """The command as it should be recorded: no venv path, no machine path."""
    return shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Sweep loading — the accuracy/size source of truth
# ---------------------------------------------------------------------------
def load_sweep(path: Path) -> list[EvalRow]:
    """Read ``sweep_dataset.csv`` back into :class:`EvalRow` objects."""
    rows: list[EvalRow] = []
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            kwargs: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "extra":
                    continue
                if key in _INT_COLUMNS:
                    kwargs[key] = int(float(value))
                elif key in _STR_COLUMNS:
                    kwargs[key] = value
                else:
                    kwargs[key] = float(value)
            rows.append(EvalRow(**kwargs))
    return rows


def check_sweep(rows: list[EvalRow], reports: list[dict[str, Any]]) -> None:
    """Prove the two artifacts describe the same measurements.

    Both the sweep and the raw capture call ``apply``/``apply_cast`` with the
    same spec, so their sizes must agree exactly (the sweep rounds to 6 decimals
    before writing). A mismatch means one of them saw a different model, spec or
    harness — every table below would then be a blend of two runs, so this fails
    instead of publishing.
    """
    problems = check_baseline(rows)
    if problems:
        raise ValueError("sweep disagrees with BASELINE_MCC: " + "; ".join(problems))

    sweep_keys = {(row.model, row.precision) for row in rows}
    raw_keys = {(entry["model"], entry["precision"]) for entry in reports}
    if sweep_keys != raw_keys:
        raise ValueError(
            "sweep and raw capture cover different measurements: "
            f"sweep-only={sorted(sweep_keys - raw_keys)} raw-only={sorted(raw_keys - sweep_keys)}"
        )

    by_key = {(entry["model"], entry["precision"]): entry for entry in reports}
    for row in rows:
        entry = by_key[(row.model, row.precision)]
        if round(entry["total_bytes"] / 1e6, 6) != row.total_mb:
            raise ValueError(
                f"{row.model}/{row.precision}: total {entry['total_bytes'] / 1e6} MB "
                f"in the raw capture vs {row.total_mb} MB in the sweep"
            )
        if round(entry["compression"], 4) != row.compression:
            raise ValueError(
                f"{row.model}/{row.precision}: compression {entry['compression']} vs "
                f"{row.compression}"
            )


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _base_columns(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": entry["model"],
        "precision": entry["precision"],
        "bits": entry["bits"],
        "dataset": entry["dataset"],
        "seed": entry["seed"],
    }


def size_table(rows: list[EvalRow], reports: list[dict[str, Any]], command: str) -> pd.DataFrame:
    """Size and compression per (model, precision)."""
    table = []
    for entry in reports:
        base = _base_columns(entry)
        total_params_with_buffers = entry["n_params_total"] + entry["buffer_bytes"] // 4
        n_quantized = sum(record["n_values"] for record in entry["records"])
        table.append(
            {
                **base,
                "spec": entry["spec"],
                "n_params": entry["n_params_total"],
                "fp32_weight_mb": round(entry["fp32_weight_bytes"] / 1e6, 6),
                "payload_mb": round(entry["packed_bytes"] / 1e6, 6),
                "metadata_mb": round((entry["quantized_bytes"] - entry["packed_bytes"]) / 1e6, 6),
                "quantized_mb": round(entry["quantized_bytes"] / 1e6, 6),
                "skipped_mb": round(entry["skipped_param_bytes"] / 1e6, 6),
                "buffers_mb": round(entry["buffer_bytes"] / 1e6, 6),
                "total_mb": round(entry["total_bytes"] / 1e6, 6),
                "compression": round(entry["compression"], 6),
                "bits_per_quantized_param": round(entry["quantized_bytes"] * 8 / n_quantized, 4),
                "bytes_per_param_incl_extras": round(
                    entry["total_bytes"] / total_params_with_buffers, 4
                ),
                "max_weight_error": round(entry["max_error"], 6),
                "artifact": ("results/quant/sweep_dataset.csv; results/quant/quant_reports.json"),
                "generated_by": command,
            }
        )
    return pd.DataFrame(table)


def accuracy_table(rows: list[EvalRow], command: str) -> pd.DataFrame:
    """Accuracy metrics per (model, precision), with the fp32 delta computed here."""
    fp32 = {row.model: row.mcc for row in rows if row.precision == "fp32"}
    table = []
    for row in rows:
        table.append(
            {
                "model": row.model,
                "precision": row.precision,
                "bits": row.bits,
                "dataset": row.dataset,
                "seed": row.seed,
                "n_test": row.n_test,
                "mcc": row.mcc,
                "accuracy": row.accuracy,
                "f1_macro": row.f1_macro,
                "mcc_delta_vs_fp32": row.mcc - fp32[row.model],
                "max_weight_error": row.max_weight_error,
                "artifact": "results/quant/sweep_dataset.csv",
                "generated_by": command,
            }
        )
    return pd.DataFrame(table)


def coverage_table(reports: list[dict[str, Any]], command: str) -> pd.DataFrame:
    """What fraction of parameters was quantized, and what was skipped and why.

    Two row kinds in one tidy table (``scope``): ``summary`` carries the
    fractions, ``tensor`` names each parameter that was left unquantized with the
    reason. Long format because the figures subset on ``scope``; a wide summary
    would hide the skipped tensors, which are the finding.
    """
    table = []
    for entry in reports:
        base = _base_columns(entry)
        n_total = entry["n_params_total"]
        tensors_total = len(entry["records"]) + len(entry["skipped"])
        if entry["kind"] == "cast":
            # A dtype cast stores *every* parameter at the narrower width,
            # including the 1-D norms and biases. Nothing is skipped, so the
            # coverage is 1.0; ``records`` still holds the matrices the
            # per-layer table needs.
            n_quantized, n_skipped = n_total, 0
            frac_bytes = 1.0
        else:
            n_quantized = sum(record["n_values"] for record in entry["records"])
            n_skipped = sum(item["n_values"] for item in entry["skipped"])
            if n_quantized + n_skipped != n_total:
                raise ValueError(
                    f"{entry['model']}/{entry['precision']}: "
                    f"{n_quantized} quantized + {n_skipped} skipped != {n_total} parameters"
                )
            frac_bytes = (
                round(entry["quantized_bytes"] / entry["total_bytes"], 6)
                if entry["total_bytes"]
                else 0.0
            )
        table.append(
            {
                **base,
                "scope": "summary",
                "scheme_kind": entry["kind"],
                "tensor": "",
                "tensor_class": "",
                "role": "",
                "quantized": True,
                "reason": "",
                "n_values": n_total,
                "bytes_fp32": n_total * 4,
                "bytes_total": entry["total_bytes"],
                "n_params_total": n_total,
                "n_params_quantized": n_quantized,
                "n_params_skipped": n_skipped,
                "n_tensors_total": tensors_total,
                "n_tensors_quantized": len(entry["records"]),
                "n_tensors_skipped": len(entry["skipped"]),
                "frac_params_quantized": round(n_quantized / n_total, 6) if n_total else 0.0,
                "frac_bytes_quantized": frac_bytes,
                "artifact": "results/quant/quant_reports.json",
                "generated_by": command,
            }
        )
        for item in entry["skipped"]:
            table.append(
                {
                    **base,
                    "scope": "tensor",
                    "scheme_kind": entry["kind"],
                    "tensor": item["name"],
                    "tensor_class": item["tensor_class"],
                    "role": item["role"],
                    "quantized": False,
                    "reason": item["reason"],
                    "n_values": item["n_values"],
                    "bytes_fp32": item["n_values"] * 4,
                    "bytes_total": "",
                    "n_params_total": n_total,
                    "n_params_quantized": n_quantized,
                    "n_params_skipped": n_skipped,
                    "n_tensors_total": tensors_total,
                    "n_tensors_quantized": len(entry["records"]),
                    "n_tensors_skipped": len(entry["skipped"]),
                    "frac_params_quantized": round(n_quantized / n_total, 6) if n_total else 0.0,
                    "frac_bytes_quantized": frac_bytes,
                    "artifact": "results/quant/quant_reports.json",
                    "generated_by": command,
                }
            )
    return pd.DataFrame(table)


def layers_table(reports: list[dict[str, Any]], command: str) -> pd.DataFrame:
    """Per-layer quantized bytes, so a figure can show where the bytes went."""
    table = []
    for entry in reports:
        base = _base_columns(entry)
        for record in entry["records"]:
            fp32_bytes = record["n_values"] * 4
            total_bytes = record["payload_bytes"] + record["metadata_bytes"]
            table.append(
                {
                    **base,
                    "layer": record["name"],
                    "tensor_class": record["tensor_class"],
                    "role": record["role"],
                    "shape": "x".join(str(dim) for dim in record["shape"]),
                    "n_values": record["n_values"],
                    "payload_bytes": record["payload_bytes"],
                    "metadata_bytes": record["metadata_bytes"],
                    "total_bytes": total_bytes,
                    "fp32_bytes": fp32_bytes,
                    "bits_per_value": round(total_bytes * 8 / record["n_values"], 4)
                    if record["n_values"]
                    else 0.0,
                    "compression": round(fp32_bytes / total_bytes, 6) if total_bytes else 1.0,
                    "max_abs_err": record["reconstruction_error"],
                    "artifact": "results/quant/quant_reports.json",
                    "generated_by": command,
                }
            )
    return pd.DataFrame(table)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def _device_and_cpu(device: Any) -> tuple[str, str]:
    if isinstance(device, dict):
        name = device.get("hostname") or device.get("ssh_alias") or "unknown"
        return str(name), str(device.get("cpu", ""))
    return str(device), ""


def _bench_row(
    device: Any,
    bench: dict[str, Any],
    *,
    runtime: str,
    source: str,
    note: str,
    device_cpu: str = "",
) -> dict[str, Any]:
    name, cpu = _device_and_cpu(device)
    return {
        "device": name,
        "device_cpu": device_cpu or cpu,
        "runtime": runtime,
        "model": bench.get("model", ""),
        "precision": bench.get("precision") or bench.get("quantization", ""),
        "dataset": bench.get("dataset", BASELINE_DATASET),
        "seed": bench.get("seed", 42),
        "iterations": bench.get("iterations", ""),
        "warmup": bench.get("warmup", ""),
        "threads": bench.get("threads", ""),
        "median_ms": bench.get("median_ms", ""),
        "p95_ms": bench.get("p95_ms", ""),
        "min_ms": bench.get("min_ms", ""),
        "max_ms": bench.get("max_ms", ""),
        "throughput_per_s": bench.get("throughput_per_s", ""),
        "artifact": source,
        "note": note,
        "generated_by": "",
    }


def latency_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """One row per (device, model, precision) found in the benchmark artifacts."""
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = json.loads(Path(path).read_text())
        rows.extend(_rows_from_payload(payload, str(path)))
    return rows


def _rows_from_payload(payload: Any, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for item in payload for row in _rows_from_payload(item, source)]
    if not isinstance(payload, dict):
        return []

    if "benchmark" in payload:
        bench = payload["benchmark"]
        runtime = str(payload.get("build", {}).get("target", ""))
        note = str(payload.get("note") or _SYNTHETIC_NOTE)
        out = [_bench_row(payload.get("device"), bench, runtime=runtime, source=source, note=note)]
        reference = payload.get("reference_dev_machine")
        if reference:
            # The reference block carries only its own measurements (median and
            # throughput). Merging it over ``bench`` handed it the router's
            # dispersion columns and the router's target label, which is how a
            # reference row ended up with min_ms above its own median. Take the
            # context from ``bench`` and nothing else.
            out.append(
                _bench_row(
                    {
                        "hostname": socket.gethostname(),
                        "cpu": reference.get("cpu") or _cpu_name(),
                    },
                    {
                        key: bench[key]
                        for key in (
                            "model",
                            "precision",
                            "dataset",
                            "seed",
                            "iterations",
                            "warmup",
                        )
                        if key in bench
                    }
                    | {
                        "median_ms": reference.get("median_ms", ""),
                        "throughput_per_s": reference.get("throughput_per_s", ""),
                    },
                    runtime=reference.get("runtime") or f"native {platform.machine()}",
                    source=source,
                    note=(
                        "reference_dev_machine in the artifact (build host); the "
                        "reference records only median/throughput, so no dispersion "
                        "is claimed for it"
                    ),
                )
            )
        return out

    if "median_ms" not in payload:
        print(f"  skipped {source}: no benchmark fields")
        return []
    return [
        _bench_row(
            payload.get("device", payload.get("device_name", "unknown")),
            payload,
            runtime=str(payload.get("runtime", "")),
            source=source,
            note=str(payload.get("note") or _SYNTHETIC_NOTE),
        )
    ]


def device_json_paths(out_dir: Path) -> list[Path]:
    """Every JSON in ``out_dir`` that is not one of this module's own artifacts."""
    return [path for path in sorted(Path(out_dir).glob("*.json")) if path.name not in _LATENCY_SKIP]


@torch.no_grad()
def measure_reference_latency(
    model_name: str,
    precision: str,
    *,
    dataset: Path | None = None,
    seed: int = 42,
    iterations: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Time the torch model on one dataset row.

    This is the *reference* path, not the accelerated one: torch reconstructs
    fp32 weights from the codes, so the timing is near-invariant across
    precisions. It is recorded as ``runtime=torch-cpu`` precisely so a reader
    cannot mistake it for a kernel speedup.
    """
    path = dataset or resolve_path(BASELINE_DATASET)
    X, _, _, _ = load_split(model_name, path, seed)
    model = reg.load(reg.by_name(model_name), strict=True)
    if precision in FLOAT_PRECISIONS:
        model, _ = apply_cast(model, precision)
    else:
        model, _ = apply(model, resolve_spec(precision))
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(X[:1]))

    for _ in range(warmup):
        model(x)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(x)
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()

    row = _bench_row(
        {"hostname": socket.gethostname(), "cpu": _cpu_name()},
        {
            "model": model_name,
            "precision": precision,
            "dataset": path.name,
            "seed": seed,
            "iterations": iterations,
            "warmup": warmup,
            "threads": torch.get_num_threads(),
            "median_ms": round(samples[len(samples) // 2], 6),
            "p95_ms": round(samples[(len(samples) * 95) // 100], 6),
            "min_ms": round(samples[0], 6),
            "max_ms": round(samples[-1], 6),
            "throughput_per_s": round(1000.0 / samples[len(samples) // 2], 4),
        },
        runtime="torch-cpu (reference; dequantizes to fp32)",
        source="measured by this command (--latency-iters)",
        note="batch-1 forward on one dataset row; reference path, not a kernel speedup",
    )
    return row


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def latency_table(rows: list[dict[str, Any]], command: str) -> pd.DataFrame:
    table = [{**row, "generated_by": command} for row in rows]
    return pd.DataFrame(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build_tables(
    out_dir: Path,
    *,
    sweep: Path = SWEEP_CSV,
    raw: Path = RAW_JSON,
    latency: list[dict[str, Any]] | None = None,
    command: str | None = None,
) -> dict[str, Path]:
    """Write the five CSVs plus the manifest that ties them to their inputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = command or recorded_command()

    payload = json.loads(Path(raw).read_text())
    reports = payload["reports"]
    rows = load_sweep(sweep)
    check_sweep(rows, reports)

    tables = {
        "size": size_table(rows, reports, command),
        "accuracy": accuracy_table(rows, command),
        "coverage": coverage_table(reports, command),
        "latency": latency_table(latency or [], command),
        "layers": layers_table(reports, command),
    }

    # A latency table holding only device artifacts is reproducible; one with a
    # row timed on this machine (e.g. ``measured by this command``) is not.
    varying = (
        {"latency"}
        if any(not Path(str(row.get("artifact", ""))).exists() for row in (latency or []))
        else set()
    )

    written: dict[str, Path] = {}
    for name, frame in tables.items():
        id_columns = ["model", "precision", "dataset", "seed"]
        if frame.empty or frame[id_columns].isna().any().any():
            raise ValueError(
                f"{name} table has {len(frame)} rows but is missing model/precision/dataset/seed; "
                "refusing to write a row that cannot be traced"
            )
        path = out_dir / TABLE_FILES[name]
        frame.to_csv(path, index=False)
        written[name] = path
        print(f"wrote {path}  ({len(frame)} rows, {len(frame.columns)} cols)")

    manifest = {
        "command": command,
        "reproduce": f"uv run {command}",
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "inputs": {str(path): _md5(path) for path in (sweep, raw) if Path(path).exists()},
        "tables": {
            name: (
                {
                    "path": str(path),
                    "rows": len(tables[name]),
                    # A table holding rows timed on the machine running this
                    # command has no reproducible hash; say so rather than
                    # publishing a number that is true for exactly one run.
                    "md5": None,
                    "varying": "machine-timed rows; regenerate with this command",
                }
                if name in varying
                else {"path": str(path), "rows": len(tables[name]), "md5": _md5(path)}
            )
            for name, path in written.items()
        },
        "latency_artifacts": sorted(
            {
                row["artifact"]
                for row in (latency or [])
                if row.get("artifact") and Path(row["artifact"]).exists()
            }
        ),
    }
    manifest_path = out_dir / MANIFEST_JSON.name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return written
