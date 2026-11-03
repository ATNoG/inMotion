"""Replay: subset selection, stream parsing and the on-device comparison (fake runtime)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from inmotion.edge import config, contracts, replay

_FAKE_BINARY = """#!/bin/sh
case "$1" in
  check) echo "cases      : 16"; echo "PARITY OK"; exit 0 ;;
  stream)
    input=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --input) input=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    n=0
    while IFS= read -r line; do
      n=$((n + 1))
      echo "{\\"sample\\":$n,\\"class\\":0,\\"label\\":\\"AA\\",\\"confidence\\":0.5}"
    done < "$input"
    exit 0 ;;
  *) echo "usage: $1 --artifact DIR"; exit 2 ;;
esac
"""


@pytest.fixture
def fake_runtime(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "inmotion-infer"
    binary.parent.mkdir(parents=True)
    binary.write_text(_FAKE_BINARY)
    binary.chmod(0o755)
    return binary


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts" / "lejepa-fp32"
    directory.mkdir(parents=True)
    (directory / contracts.ARTIFACT_HASH_FILE).write_bytes(b"weights")
    (directory / "meta.json").write_text("{}")
    return directory


@pytest.fixture
def windows(tmp_path: Path) -> Path:
    root = tmp_path / "windows"
    directory = root / "lejepa"
    directory.mkdir(parents=True)
    directory.joinpath("windows.txt").write_text("1 2 3 4\n5 6 7 8\n")
    directory.joinpath("labels.txt").write_text("0\n0\n")
    directory.joinpath("classes.json").write_text(json.dumps(["AA", "AB", "BA", "BB"]))
    return root


@pytest.fixture
def local(tmp_path: Path) -> object:
    from inmotion.edge import transport

    return transport.LocalTransport(root=str(tmp_path / "device"))


def _profile() -> config.Profile:
    return dataclasses.replace(
        config.PROFILES["smoke"],
        models=("lejepa",),
        precisions=("fp32",),
        replay_samples={"local": 8},
    )


def test_stratified_subset_is_exact_and_deterministic() -> None:
    labels = [index % 4 for index in range(703)]
    first = replay.stratified_subset(labels, 200)
    second = replay.stratified_subset(labels, 200)
    assert first == second
    assert len(first) == 200
    assert len(set(first)) == 200
    # Every class is represented, proportionally.
    counts = {label: sum(1 for index in first if labels[index] == label) for label in range(4)}
    assert set(counts) == set(range(4))
    assert all(48 <= count <= 52 for count in counts.values())


def test_stratified_subset_keeps_everything_when_it_can() -> None:
    labels = [0, 1, 0, 1]
    assert replay.stratified_subset(labels, 0) == [0, 1, 2, 3]
    assert replay.stratified_subset(labels, 99) == [0, 1, 2, 3]


def test_load_windows_reads_the_contract_layout(windows: Path) -> None:
    rows, labels, classes = replay.load_windows(windows, "lejepa")
    assert rows == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert labels == [0, 0]
    assert classes == ["AA", "AB", "BA", "BB"]


def test_load_windows_rejects_a_length_mismatch(windows: Path) -> None:
    (windows / "lejepa" / "labels.txt").write_text("0\n")
    with pytest.raises(ValueError, match="labels"):
        replay.load_windows(windows, "lejepa")


def test_parse_stream_orders_by_sample_and_ignores_noise() -> None:
    stdout = (
        'note\n{"sample":2,"class":3,"confidence":0.4}\n'
        '{"sample":1,"class":0,"confidence":0.9}\nnot json\n'
    )
    assert replay.parse_stream(stdout) == [0, 3]


def test_agreement_and_mcc_on_degenerate_inputs() -> None:
    assert replay.agreement([0, 1, 1], [0, 1, 1]) == 1.0
    assert replay.agreement([0, 1, 1], [0, 0, 0]) == pytest.approx(1 / 3)
    assert replay.mcc([0, 0], [0, 0]) == replay.MCC_UNAVAILABLE


def test_python_predictions_returns_none_on_an_unreadable_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "nowhere"
    directory.mkdir()
    assert replay.python_predictions(directory, [[0.0] * 4]) is None


def test_run_writes_a_replay_row_with_the_golden_gate(
    fake_runtime: Path, artifact: Path, windows: Path, local: object, tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    path = replay.run(
        local,
        artifact.parent,
        windows,
        tables,
        tmp_path / "raw",
        _profile(),
        binary=fake_runtime,
    )
    assert path == tables / "replay.csv"
    rows = config.read_table(tables / "replay.csv", "replay")
    assert len(rows) == 1
    row = rows[0]
    assert row["device"] == "local"
    assert row["samples"] == "2"
    assert row["status"] == "ok"
    assert row["parity_ok"] == "True"
    assert row["artifact_sha256"] == hashlib.sha256(b"weights").hexdigest()
    # No meta reader for this fixture, so the Python cross-check is unavailable,
    # not zero: the golden gate is what parity rests on.
    assert row["mcc_python"] == "-2.0"
    assert row["agreement"] == "-1.0"

    audit = json.loads((tmp_path / "raw" / "replay" / "local-lejepa-fp32.json").read_text())
    assert audit["indices"] == [0, 1]
    assert audit["golden_ok"] is True


def test_run_can_skip_a_profile_with_no_samples(
    fake_runtime: Path, artifact: Path, windows: Path, local: object, tmp_path: Path
) -> None:
    profile = dataclasses.replace(_profile(), replay_samples={})
    tables = tmp_path / "tables"
    replay.run(
        local, artifact.parent, windows, tables, tmp_path / "raw", profile, binary=fake_runtime
    )
    rows = config.read_table(tables / "replay.csv", "replay")
    assert rows == []
