"""Replay the test windows on the device and compare with the Python model.

Parity has two halves. ``check --golden`` proves the runtime computes the same
logits as the exporter on sixteen golden windows, which is the strong check: it
is the same tensors, at the same tolerance, as the port itself. Replaying the
whole test split then proves that the *pipeline* agrees sample by sample: the
same artifact, the same windows, the same labels, on the router.

The per-sample comparison needs Python predictions for the *same artifact*, so
they are rebuilt per (model, precision) from that artifact's own weights
(dequantize, rebuild the state dict, load, forward). That is exactly the
accuracy workstream's method, so ``mcc_python`` on a 200-sample device-2 subset
is the same number ``accuracy.csv`` records for that precision, and
``agreement`` is device-vs-same-precision rather than device-vs-fp32. An fp32
agreement is kept in the raw audit JSON as a diagnostic only.

``max_logit_diff`` is the maximum absolute logit difference from the golden
check (``check --golden``), i.e. the quantitative half of the parity claim; the
``parity_ok`` boolean is the gate. It is ``-1`` only when the check did not run
or did not print a number, never as a placeholder for "not measured".

Device 1 replays all 703 samples; device 2 replays a fixed stratified 200, so
the two devices are compared on the same windows every time.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from inmotion.edge import contracts, transport
from inmotion.edge.bench import ensure_remote_artifact
from inmotion.edge.config import Profile, cell, write_table

#: The strong parity gate: golden cases must all pass before a replay row is
#: allowed to say "ok".
_PARITY_MCC = 0.005
_PARITY_AGREEMENT = 0.995
_STREAM_TIMEOUT_S = 600.0


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ── windows and labels ───────────────────────────────────────────────────


def load_windows(
    windows_dir: Path | str, model: str
) -> tuple[list[list[float]], list[int], list[str]]:
    """Read ``<windows_dir>/<model>/{windows,labels}.txt`` and ``classes.json``.

    The flat directory layout is accepted too, for a single-model fixture.
    """
    root = Path(windows_dir)
    base = root / model
    if not (base / "windows.txt").is_file():
        base = root
    windows_path = base / "windows.txt"
    labels_path = base / "labels.txt"
    if not windows_path.is_file():
        raise FileNotFoundError(f"replay: no windows.txt for {model!r} under {root}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"replay: no labels.txt for {model!r} under {root}")
    windows = [
        [float(value) for value in line.split()]
        for line in windows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        int(float(line.split()[0]))
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(windows) != len(labels):
        raise ValueError(f"replay: {model}: {len(windows)} windows but {len(labels)} labels")
    if not windows:
        raise ValueError(f"replay: {model}: windows.txt is empty")
    width = len(windows[0])
    if any(len(row) != width for row in windows):
        raise ValueError(f"replay: {model}: windows do not all have {width} values")
    classes: list[str] = []
    classes_path = base / "classes.json"
    if classes_path.is_file():
        payload = json.loads(classes_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("classes", [])
        if isinstance(payload, list):
            classes = [str(item) for item in payload]
    return windows, labels, classes


def stratified_subset(labels: Sequence[int], count: int) -> list[int]:
    """A fixed stratified subset of sample indices.

    Allocation is proportional with largest remainders, and the picks inside a
    class are evenly spaced, so the 200 device-2 samples do not drift when the
    label file is regenerated with the same split.
    """
    total = len(labels)
    if count <= 0 or count >= total:
        return list(range(total))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    quotas = {label: count * len(indices) / total for label, indices in groups.items()}
    allocation = {label: int(quota) for label, quota in quotas.items()}
    remaining = count - sum(allocation.values())
    order = sorted(
        groups, key=lambda label: (quotas[label] - allocation[label], -label), reverse=True
    )
    for label in order[:remaining]:
        allocation[label] += 1
    selected: list[int] = []
    for label in sorted(groups):
        pool = groups[label]
        wanted = min(allocation[label], len(pool))
        if wanted <= 0:
            continue
        step = len(pool) / wanted
        picked = [
            pool[min(len(pool) - 1, int(round(position * step)))] for position in range(wanted)
        ]
        selected.extend(dict.fromkeys(picked))
    if len(selected) < count:
        leftover = [index for index in range(total) if index not in set(selected)]
        selected.extend(leftover[: count - len(selected)])
    return sorted(selected)


# ── Python side ──────────────────────────────────────────────────────────


def python_predictions(
    artifact_dir: Path | str, windows: Sequence[Sequence[float]]
) -> list[int] | None:
    """Predict the windows with the artifact's own weights, or ``None``.

    Imported lazily: replay on a router-only machine still works without torch,
    it just cannot cross-check sample by sample (the golden check still runs).
    """
    try:
        import numpy as np
        import torch
        from safetensors.numpy import load_file

        from inmotion.models.adopt import resolve_model
        from inmotion.models.factories import build
        from inmotion.quant.export import dequantize_artifact, rebuild_state_dict
    except ImportError as exc:
        _say(f"replay: Python predictions unavailable ({exc})")
        return None

    directory = Path(artifact_dir)
    try:
        # `dequantize_artifact` is the artifact's single inverse (bit-unpacking,
        # folded bias, per-group expansion) and is shared with the accuracy
        # scoring. Reusing it is what makes `mcc_python` the same number
        # `accuracy.csv` records for that precision.
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        arrays = load_file(str(directory / contracts.ARTIFACT_HASH_FILE))
        weights = dequantize_artifact(meta, arrays)
    except Exception as exc:  # noqa: BLE001 - an optional cross-check must not abort a replay
        _say(f"replay: cannot read {directory} in Python ({exc})")
        return None

    try:
        try:
            spec = resolve_model(str(meta["model"]))
        except KeyError:
            spec = resolve_model(str(meta["factory"]))
        model = build(spec.factory, spec.arch).eval()
        model.load_state_dict(rebuild_state_dict(model, weights, spec.prefix), strict=True)
        x = np.asarray(windows, dtype=np.float32)
        seq_len = int(meta["seq_len"])
        channels = int(meta["in_channels"])
        expected = len(windows) * seq_len * channels
        if x.size != expected:
            _say(
                f"replay: {x.size} window values do not match the artifact "
                f"({len(windows)} samples x {seq_len} x {channels})"
            )
            return None
        x = x.reshape(len(windows), seq_len, channels)
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()
        return [int(row.argmax()) for row in logits]
    except Exception as exc:  # noqa: BLE001 - an optional cross-check must not abort a replay
        _say(f"replay: Python predictions failed ({type(exc).__name__}: {exc})")
        return None


#: MCC is defined on [-1, 1], so ``-1`` cannot be an "unavailable" sentinel: it
#: is a perfect anti-correlation. Degenerate inputs (one class only) use ``-2``,
#: which is outside the metric's range and is documented in the table.
MCC_UNAVAILABLE = -2.0


def mcc(truth: Sequence[int], predicted: Sequence[int]) -> float:
    """Matthews correlation; ``MCC_UNAVAILABLE`` for a degenerate input.

    Using ``-1`` here would collide with a legitimate perfect
    anti-correlation, and q1 predictions are genuinely negative-MCC (lejepa
    -0.0013), so every guard below compares against ``MCC_UNAVAILABLE``.
    """
    from sklearn.metrics import matthews_corrcoef

    if len(set(truth)) < 2 or len(set(predicted)) < 2:
        return MCC_UNAVAILABLE
    return float(matthews_corrcoef(list(truth), list(predicted)))


def agreement(truth: Sequence[int], predicted: Sequence[int]) -> float:
    """Share of samples where two label sequences agree."""
    if not truth:
        return -1.0
    return sum(1 for a, b in zip(truth, predicted, strict=True) if a == b) / len(truth)


# ── device side ──────────────────────────────────────────────────────────


def parse_stream(stdout: str) -> list[int]:
    """Class indices from ``stream --json`` lines, sorted by sample number."""
    records: list[tuple[int, int]] = []
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "class" not in payload:
            continue
        sample = int(payload.get("sample") or len(records) + 1)
        records.append((sample, int(payload["class"])))
    records.sort()
    return [value for _sample, value in records]


def check_golden(
    device: transport.DeviceTransport, binary: str, remote_artifact: str
) -> tuple[bool, str, float]:
    """Run ``check --golden``; return (passed, failure detail, max |logit diff|).

    The runtime prints ``max |diff| : <float>``; that number is the quantitative
    parity result and is what ``replay.csv`` carries. It is ``-1`` only when the
    check failed before printing it.
    """
    result = device.run(
        f"{binary} check --artifact {transport.shell_quote(remote_artifact)} "
        f"--golden {transport.shell_quote(remote_artifact)}",
        timeout=300.0,
    )
    ok = result.returncode == 0 and "PARITY OK" in result.stdout
    match = re.search(r"max \|diff\|\s*:\s*([0-9.eE+-]+)", result.stdout)
    max_diff = float(match.group(1)) if match else -1.0
    detail = "" if ok else (result.stderr.strip() or result.stdout.strip())[:200]
    return ok, detail, max_diff


def _write_subset(path: Path, windows: Sequence[Sequence[float]], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(" ".join(repr(float(value)) for value in windows[index]) + "\n")
    tmp.replace(path)


def _accuracy_mcc(path: Path, model: str, precision: str) -> float:
    """``accuracy.csv``'s MCC for this cell, so replay can prove it agrees."""
    if not path.is_file():
        return MCC_UNAVAILABLE
    import csv

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("model") == model and row.get("precision") == precision:
                    return float(row.get("mcc") or MCC_UNAVAILABLE)
    except (OSError, ValueError):
        return MCC_UNAVAILABLE
    return MCC_UNAVAILABLE


# ── the run ──────────────────────────────────────────────────────────────


def run(
    device: transport.DeviceTransport,
    artifacts_dir: Path | str,
    windows_dir: Path | str,
    tables_dir: Path | str,
    raw_dir: Path | str,
    profile: Profile,
    *,
    binary: Path | str | None = None,
    accuracy_csv: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Replay the profile's matrix on ``device``; write ``replay.csv``.

    ``accuracy_csv`` is where the accuracy table lives when ``tables_dir`` is a
    per-device staging directory (the campaign passes the run's shared table);
    the value is used only to assert that the Python MCC here equals the one the
    accuracy workstream reported for the same artifact.
    """
    say = log or _say
    from inmotion.edge.bench import resolve_artifacts

    table_dir = Path(tables_dir)
    raw = Path(raw_dir) / "replay"
    raw.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifacts(artifacts_dir)
    wanted = [(model, precision) for model in profile.models for precision in profile.precisions]
    missing = [pair for pair in wanted if pair not in artifacts]
    if missing:
        listing = ", ".join(f"{model}-{precision}" for model, precision in missing[:8])
        raise FileNotFoundError(f"replay: no artifact for {listing} under {artifacts_dir}")

    remote_binary = f"{device.root.rstrip('/')}/bin/inmotion-infer"
    if binary is not None:
        from inmotion.edge.bench import deploy_binary

        remote_binary = deploy_binary(device, Path(binary), log=say)

    count = profile.replay_count(device.key)
    if count <= 0:
        say(f"replay: profile {profile.name} has no samples for {device.key}; writing header only")
        path = write_table(table_dir / "replay.csv", "replay", [])
        return path

    rows: list[dict[str, object]] = []
    remote_windows: dict[str, str] = {}
    fp32_predictions: dict[str, list[int] | None] = {}
    for model in profile.models:
        windows, labels, _classes = load_windows(windows_dir, model)
        indices = stratified_subset(labels, count)
        subset_labels = [labels[index] for index in indices]
        local_subset = raw / f"{device.key}-{model}-windows.txt"
        _write_subset(local_subset, windows, indices)
        remote_subset = f"{device.root.rstrip('/')}/replay/{device.key}-{model}-windows.txt"
        device.mkdir(f"{device.root.rstrip('/')}/replay")
        device.upload(local_subset, remote_subset)
        remote_windows[model] = remote_subset
        subset_windows = [windows[index] for index in indices]
        for precision in profile.precisions:
            local_artifact = artifacts[(model, precision)]
            artifact_sha = transport.collect_artifact_sha256(local_artifact)
            remote_artifact = ensure_remote_artifact(device, local_artifact, log=say)
            golden_ok, golden_detail, max_logit_diff = check_golden(
                device, remote_binary, remote_artifact
            )
            stream = device.run(
                f"{remote_binary} stream --artifact "
                f"{transport.shell_quote(remote_artifact)} --input "
                f"{transport.shell_quote(remote_windows[model])} --json",
                timeout=_STREAM_TIMEOUT_S,
            )
            device_predictions = parse_stream(stream.stdout)
            status = "ok"
            if stream.returncode != 0:
                status = f"stream-error:{(stream.stderr.strip() or stream.stdout.strip())[:180]}"
            elif len(device_predictions) != len(indices):
                status = f"sample-mismatch:{len(device_predictions)}/{len(indices)}"
            elif not golden_ok:
                status = "ok:no-golden"
                if golden_detail:
                    say(f"replay: {model}-{precision}: golden failed: {golden_detail}")

            usable = len(device_predictions) == len(indices) and not status.startswith(
                "stream-error"
            )
            mcc_device = mcc(subset_labels, device_predictions) if usable else MCC_UNAVAILABLE
            # Same artifact, same windows: this is the comparison `accuracy.csv`
            # makes, so its MCC and ours must agree for the same precision.
            python = python_predictions(local_artifact, subset_windows)
            if python is None:
                _say(
                    f"replay: {device.key} {model}-{precision}: "
                    "no Python predictions (golden gate only)"
                )
            if python is not None and len(python) == len(indices):
                mcc_python = mcc(subset_labels, python)
                agree = agreement(python, device_predictions) if usable else -1.0
                measurable = mcc_python > MCC_UNAVAILABLE and mcc_device > MCC_UNAVAILABLE
                delta = mcc_device - mcc_python if (usable and measurable) else MCC_UNAVAILABLE
            else:
                mcc_python = MCC_UNAVAILABLE
                agree = -1.0
                delta = MCC_UNAVAILABLE
            if usable and golden_ok:
                if agree >= 0 and delta > MCC_UNAVAILABLE:
                    parity_ok = abs(delta) <= _PARITY_MCC and agree >= _PARITY_AGREEMENT
                elif agree >= 0:
                    # MCC is undefined on both sides (a single predicted class);
                    # the label-by-label agreement is the stronger proof anyway.
                    parity_ok = agree >= _PARITY_AGREEMENT
                else:
                    parity_ok = True  # the golden vectors are the available proof
            else:
                parity_ok = False
            if precision == "fp32":
                fp32_predictions[model] = python
            fp32 = fp32_predictions.get(model)
            fp32_agreement = (
                agreement(fp32, device_predictions)
                if fp32 is not None and usable and len(fp32) == len(device_predictions)
                else -1.0
            )
            reference = Path(accuracy_csv) if accuracy_csv else Path(tables_dir) / "accuracy.csv"
            accuracy_mcc = _accuracy_mcc(reference, model, precision)

            rows.append(
                {
                    "device": device.key,
                    "model": model,
                    "precision": precision,
                    "samples": len(indices),
                    "agreement": agree,
                    "mcc_python": mcc_python,
                    "mcc_device": mcc_device,
                    "mcc_delta": delta,
                    "max_logit_diff": max_logit_diff,
                    "parity_ok": parity_ok,
                    "artifact_sha256": artifact_sha,
                    "status": status,
                }
            )
            audit = {
                "device": device.key,
                "model": model,
                "precision": precision,
                "samples": len(indices),
                "indices": indices,
                "labels": subset_labels,
                "device_predictions": device_predictions,
                "python_predictions": python,
                "agreement_vs_fp32": fp32_agreement,
                "accuracy_csv_mcc": accuracy_mcc,
                "golden_ok": golden_ok,
                "golden_detail": golden_detail,
                "max_logit_diff": max_logit_diff,
                "status": status,
            }
            if (
                accuracy_mcc > MCC_UNAVAILABLE
                and mcc_python > MCC_UNAVAILABLE
                and abs(accuracy_mcc - mcc_python) > 0.001
            ):
                say(
                    f"replay: warning: {model}-{precision}: Python MCC {mcc_python:.6f} "
                    f"differs from accuracy.csv {accuracy_mcc:.6f}"
                )
            (raw / f"{device.key}-{model}-{precision}.json").write_text(
                json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
            )
            say(
                f"replay: {device.key} {model}-{precision}: n={len(indices)} "
                f"agreement={cell(agree)} dMCC={cell(delta)} status={status}"
            )

    path = write_table(table_dir / "replay.csv", "replay", rows)
    say(f"replay: {device.key}: {path} ({len(rows)} rows)")
    return path
