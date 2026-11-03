"""The canonical accuracy path: export, artifact scoring, windows.

These run against one real registered checkpoint (``lejepa``) and three
precisions, so the export/scoring code is exercised end to end without paying
for the full five-model sweep. The properties checked are the ones the campaign
depends on: the fp32 artifact reproduces ``contracts.BASELINE_MCC`` bit-for-bit,
every deployable row is scored from the artifact's own weights (with the
in-memory quantizer only as a witness), the cast rows are appended with no byte
count, and the replay windows match the scored split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from inmotion.edge import accuracy
from inmotion.edge import contracts as c

MODEL = "lejepa"
PRECISIONS = ("fp32", "q4", "q2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("edge-artifacts")
    accuracy.export_artifacts(directory, models=(MODEL,), precisions=PRECISIONS)
    return directory


@pytest.fixture(scope="module")
def tables(artifacts: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("edge-tables")
    accuracy.evaluate_artifacts(artifacts, directory, models=(MODEL,), precisions=PRECISIONS)
    return directory


# ── Artifact export ──────────────────────────────────────────────────────


def test_export_artifacts_writes_dirs_golden_and_table(artifacts: Path) -> None:
    rows = _rows(artifacts / "artifacts.csv")
    assert list(rows[0]) == list(c.COLUMNS["artifacts"])
    assert [row["precision"] for row in rows] == list(PRECISIONS)
    for row in rows:
        directory = artifacts / f"{MODEL}-{row['precision']}"
        weights = directory / c.ARTIFACT_HASH_FILE
        meta = directory / "meta.json"
        golden = directory / "golden.json"
        for path in (directory, weights, meta, golden):
            assert path.exists(), path
        assert row["status"] == "ok"
        assert row["error"] == ""
        assert row["runtime_loadable"] == "true"
        assert int(row["bytes"]) == weights.stat().st_size + meta.stat().st_size
        assert row["sha256"] == _sha256(weights)
        assert row["path"] == str(directory)
        payload = json.loads(golden.read_text())
        assert payload["tol"] == pytest.approx(2e-4)
        assert len(payload["cases"]) > 0
        # golden.json is a parity file, not part of the runtime payload.
        assert int(row["bytes"]) < sum(
            path.stat().st_size for path in directory.iterdir() if path.is_file()
        )


def test_export_artifacts_refuses_cast_precision_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown in-house scheme"):
        accuracy.export_artifacts(tmp_path, models=(MODEL,), precisions=("bf16",))
    assert not (tmp_path / f"{MODEL}-bf16").exists()


# ── Artifact scoring ────────────────────────────────────────────────────


def test_evaluate_writes_exact_schemas_and_row_counts(tables: Path) -> None:
    expected = {
        "accuracy.csv": c.COLUMNS["accuracy"],
        "per_class.csv": c.COLUMNS["per_class"],
        "confusion.csv": c.COLUMNS["confusion"],
    }
    for name, columns in expected.items():
        with (tables / name).open(newline="") as handle:
            assert tuple(next(csv.reader(handle))) == columns

    accuracy_rows = _rows(tables / "accuracy.csv")
    assert len(accuracy_rows) == len(PRECISIONS) + len(c.PRECISIONS_CAST)
    assert [row["precision"] for row in accuracy_rows] == list(PRECISIONS) + list(c.PRECISIONS_CAST)
    assert len(_rows(tables / "per_class.csv")) == len(PRECISIONS) * 4
    assert len(_rows(tables / "confusion.csv")) == len(PRECISIONS) * 16


def test_cast_rows_are_appended_without_artifacts(tables: Path) -> None:
    rows = _rows(tables / "accuracy.csv")
    cast = [row for row in rows if row["runtime_loadable"] == "false"]
    assert [row["precision"] for row in cast] == list(c.PRECISIONS_CAST)
    for row in cast:
        assert int(row["artifact_bytes"]) == -1
        assert row["artifact_sha256"] == ""
        assert float(row["effective_bits"]) in {8.0, 16.0}
        assert 0.0 <= float(row["mcc"]) <= 1.0


def test_fp32_artifact_mcc_equals_recorded_baseline_exactly(tables: Path) -> None:
    rows = _rows(tables / "accuracy.csv")
    fp32 = next(row for row in rows if row["precision"] == "fp32")
    # The CSV stores Python's shortest round-trip repr; comparing the parsed
    # float is the exact comparison the campaign promises.
    assert float(fp32["mcc"]) == c.BASELINE_MCC[MODEL]
    assert float(fp32["delta_mcc"]) == 0.0
    assert int(fp32["n_params"]) > 0
    assert int(fp32["artifact_bytes"]) > 0
    assert len(fp32["artifact_sha256"]) == 64


def test_effective_bits_use_artifact_bytes_and_count_metadata(tables: Path) -> None:
    frame = pd.read_csv(tables / "accuracy.csv", float_precision="round_trip")
    deployable = frame[frame["runtime_loadable"].astype(str).str.lower() == "true"]
    for row in deployable.itertuples(index=False):
        expected = c.effective_bits(int(row.artifact_bytes), int(row.n_params))
        assert float(row.effective_bits) == pytest.approx(expected)
        assert float(row.effective_bits) > 0
    fp32 = deployable[deployable["precision"] == "fp32"].iloc[0]
    # Metadata is part of the artifact, so fp32 is never exactly 32 bits/param.
    assert float(fp32["effective_bits"]) > 32.0


def test_ci_bounds_enclose_point_estimates(tables: Path) -> None:
    frame = pd.read_csv(tables / "accuracy.csv", float_precision="round_trip")
    for row in frame.itertuples(index=False):
        assert float(row.mcc_lo) <= float(row.mcc) <= float(row.mcc_hi)
        assert float(row.accuracy_lo) <= float(row.accuracy) <= float(row.accuracy_hi)
        assert float(row.f1_lo) <= float(row.f1_macro) <= float(row.f1_hi)
        assert float(row.delta_lo) <= float(row.delta_mcc) <= float(row.delta_hi)


def test_per_class_and_confusion_are_consistent(tables: Path) -> None:
    per_class = pd.read_csv(tables / "per_class.csv", float_precision="round_trip")
    confusion = pd.read_csv(tables / "confusion.csv", float_precision="round_trip")
    for (model, precision), block in confusion.groupby(["model", "precision"]):
        assert int(block["count"].sum()) == c.N_TEST
        per = per_class[(per_class["model"] == model) & (per_class["precision"] == precision)]
        assert int(per["support"].sum()) == c.N_TEST
        # tp/fp/fn are derived from the same matrix, so they must agree with it.
        for _, row in per.iterrows():
            name = str(row["class"])
            tp = int(
                block[(block["true_class"] == name) & (block["predicted_class"] == name)][
                    "count"
                ].iloc[0]
            )
            assert int(row["tp"]) == tp
            assert int(row["tp"] + row["fn"]) == int(row["support"])


def test_evaluation_is_deterministic(artifacts: Path, tables: Path, tmp_path: Path) -> None:
    second = tmp_path / "tables"
    accuracy.evaluate_artifacts(artifacts, second, models=(MODEL,), precisions=PRECISIONS)
    columns = [
        "model",
        "precision",
        "mcc",
        "mcc_lo",
        "mcc_hi",
        "accuracy",
        "accuracy_lo",
        "accuracy_hi",
        "f1_macro",
        "delta_mcc",
        "delta_lo",
        "delta_hi",
    ]
    first = pd.read_csv(tables / "accuracy.csv", float_precision="round_trip")[columns]
    again = pd.read_csv(second / "accuracy.csv", float_precision="round_trip")[columns]
    pd.testing.assert_frame_equal(first, again)


def test_fp32_baseline_guard_fails_loudly(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(c.BASELINE_MCC, MODEL, 0.123)
    with pytest.raises(RuntimeError, match="baseline"):
        accuracy.evaluate_artifacts(
            artifacts, tmp_path / "tables", models=(MODEL,), precisions=("fp32",)
        )


def test_artifact_witness_guard_fails_loudly(artifacts: Path, tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifacts, copied)
    weights = copied / f"{MODEL}-q4" / c.ARTIFACT_HASH_FILE
    import numpy as np
    from safetensors.numpy import load_file, save_file

    arrays = load_file(str(weights))
    for key in list(arrays):
        if key.endswith(".scale"):
            arrays[key] = (arrays[key] * 7.0).astype(np.float32)
    save_file(arrays, str(weights))

    with pytest.raises(RuntimeError, match="disagree"):
        accuracy.evaluate_artifacts(
            copied, tmp_path / "tables", models=(MODEL,), precisions=("q4",)
        )


def test_missing_artifact_error_names_the_export_command(artifacts: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        accuracy.evaluate_artifacts(
            artifacts, tmp_path / "tables", models=(MODEL,), precisions=("q8",)
        )
    message = str(caught.value)
    assert f"{MODEL}-q8" in message
    assert "export_artifacts" in message


# ── Windows for the device replay ───────────────────────────────────────


def test_export_windows_matches_the_scored_split(tmp_path: Path) -> None:
    paths = accuracy.export_windows(tmp_path / "windows", models=(MODEL,))
    windows = paths[f"{MODEL}/windows"].read_text().splitlines()
    labels = paths[f"{MODEL}/labels"].read_text().splitlines()
    classes = json.loads(paths[f"{MODEL}/classes"].read_text())
    assert len(windows) == len(labels) == c.N_TEST
    assert len(classes) == 4
    assert all(isinstance(name, str) and name for name in classes)
    # lejepa consumes the 4-channel features: seq_len 10 * 4 channels.
    assert all(len(line.split()) == 10 * 4 for line in windows)
    values = [int(label) for label in labels]
    assert set(values) <= set(range(len(classes)))
    assert set(values) == set(range(len(classes)))  # all four classes are present


def test_export_windows_uses_each_models_channel_count(tmp_path: Path) -> None:
    # t_jepa consumes the rich 18-channel features, so its windows are wider.
    paths = accuracy.export_windows(tmp_path / "windows", models=("t_jepa",))
    windows = paths["t_jepa/windows"].read_text().splitlines()
    assert len(windows) == c.N_TEST
    assert all(len(line.split()) == 10 * 18 for line in windows)
