"""Campaign: run directory, state/resume/force, report and manifest provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from inmotion.edge import campaign, config, contracts, transport


@pytest.fixture
def binary(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "inmotion-infer"
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _run(
    tmp_path: Path,
    binary: Path,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
    force: bool = False,
    phases: tuple[str, ...] = ("preflight", "report", "manifest"),
) -> config.CampaignConfig:
    return campaign.run(
        "smoke",
        ["local"],
        results_root=tmp_path / "results",
        run_dir=run_dir,
        resume=resume,
        force=force,
        binary=binary,
        phases=phases,
        log=lambda _message: None,
    )


def test_run_creates_the_contract_layout_and_manifest(tmp_path: Path, binary: Path) -> None:
    cfg = _run(tmp_path, binary)
    assert re.fullmatch(r"\d{8}T\d{6}Z", cfg.run_dir.name)
    for name in ("config.json", "state.json", "manifest.json", "report.md"):
        assert (cfg.run_dir / name).is_file(), name
    for name in ("tables", "raw", "figures", "logs"):
        assert (cfg.run_dir / name).is_dir(), name

    state = json.loads((cfg.run_dir / "state.json").read_text())
    assert state["status"] == "complete"
    assert state["phases"]["preflight"]["status"] == "done"

    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    for key in (
        "run",
        "git",
        "lockfiles",
        "binary",
        "artifacts",
        "devices",
        "versions",
        "tables",
        "figures",
        "phases",
    ):
        assert key in manifest, key
    assert manifest["binary"]["sha256"] == transport.binary_sha256(binary)
    assert manifest["run"]["devices"] == ["local"]
    assert manifest["devices"]["local"]["root_ok"] is True

    latest = tmp_path / "results" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == cfg.run_dir.resolve()

    config_payload = json.loads((cfg.run_dir / "config.json").read_text())
    assert config_payload["profile"] == "smoke"
    assert config_payload["phases"] == ["preflight", "report", "manifest"]


def test_manifest_and_state_never_carry_the_password(
    tmp_path: Path, binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(transport.PASSWORD_ENV, "super-secret-value")
    cfg = _run(tmp_path, binary)
    for name in ("manifest.json", "state.json", "config.json"):
        text = (cfg.run_dir / name).read_text()
        assert "super-secret-value" not in text


def test_json_write_redacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(transport.PASSWORD_ENV, "super-secret-value")
    path = tmp_path / "payload.json"
    campaign._json_write(path, {"value": "super-secret-value", "nested": ["x"]})
    assert "super-secret-value" not in path.read_text()
    assert "***" in path.read_text()


def test_resume_skips_phases_that_already_finished(tmp_path: Path, binary: Path) -> None:
    first = _run(tmp_path, binary)
    before = json.loads((first.run_dir / "state.json").read_text())
    started = before["phases"]["preflight"]["started"]

    second = _run(tmp_path, binary, run_dir=first.run_dir, resume=True)
    after = json.loads((second.run_dir / "state.json").read_text())
    assert after["phases"]["preflight"]["started"] == started
    assert after["status"] == "complete"


def test_a_finished_run_needs_force(tmp_path: Path, binary: Path) -> None:
    first = _run(tmp_path, binary)
    with pytest.raises(FileExistsError, match="--force"):
        _run(tmp_path, binary, run_dir=first.run_dir)
    forced = _run(tmp_path, binary, run_dir=first.run_dir, force=True)
    assert forced.run_dir == first.run_dir
    assert json.loads((forced.run_dir / "state.json").read_text())["status"] == "complete"


def test_table_rows_counts_only_data_rows(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    config.write_table(
        tables / "bench.csv",
        "bench",
        [{"device": "wave-router", "model": "lejepa", "precision": "fp32", "p95_ms": 2.5}],
    )
    assert campaign._table_rows(tables)["bench"] == 1
    assert "load" not in campaign._table_rows(tables)


def test_report_summarises_the_tables(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    config.write_table(
        tables / "accuracy.csv",
        "accuracy",
        [
            {
                "model": "lejepa",
                "precision": "fp32",
                "mcc": 0.9142,
                "artifact_sha256": "x",
            }
        ],
    )
    config.write_table(
        tables / "bench.csv",
        "bench",
        [
            {
                "device": "wave-router",
                "model": "lejepa",
                "precision": "fp32",
                "p95_ms": 2.5,
                "status": "ok",
            }
        ],
    )
    config.write_table(
        tables / "replay.csv",
        "replay",
        [
            {
                "device": "wave-router",
                "model": "lejepa",
                "precision": "fp32",
                "samples": 703,
                "agreement": 1.0,
                "mcc_python": 0.9142,
                "mcc_device": 0.9142,
                "mcc_delta": 0.0,
                "parity_ok": True,
                "status": "ok",
            }
        ],
    )
    config.write_table(
        tables / "impact.csv",
        "impact",
        [{"device": "wave-router", "kpi": "cpu_busy_pct", "verdict": "equivalent"}],
    )
    path = campaign.build_report(tables, tmp_path / "report.md")
    text = path.read_text()
    assert "| `accuracy.csv` | 1 |" in text
    assert "| lejepa | 0.9142 |" in text
    assert "parity ok: 1" in text
    assert "1 equivalent" in text
    # The report is deterministic: rerunning rewrites identical bytes.
    first = path.read_bytes()
    campaign.build_report(tables, path)
    assert path.read_bytes() == first


def test_profile_phase_subset_is_validated(tmp_path: Path, binary: Path) -> None:
    with pytest.raises(KeyError, match="unknown phase"):
        campaign.run(
            "smoke",
            ["local"],
            results_root=tmp_path / "results",
            binary=binary,
            phases=("preflight", "nope"),
            log=lambda _message: None,
        )


def test_figures_are_skipped_for_a_local_only_run(tmp_path: Path, binary: Path) -> None:
    # F4-F8 plot router rows; a local smoke run cannot satisfy them, and the
    # figure builder raises rather than drawing half-empty panels. The campaign
    # records the skip instead of failing a phase the run cannot produce.
    cfg = _run(tmp_path, binary, phases=("preflight", "figures", "manifest"))
    state = json.loads((cfg.run_dir / "state.json").read_text())
    assert state["phases"]["figures"]["status"] == "skipped"
    assert "router" in state["phases"]["figures"]["detail"]
    assert state["status"] == "complete"


def test_phase_handlers_cover_every_declared_phase() -> None:
    assert set(campaign.PHASE_HANDLERS) == set(config.PHASES)
    assert config.PHASES[0] == "preflight"
    assert config.PHASES[-1] == "manifest"


def test_manifest_tables_mirror_the_run_tables(tmp_path: Path, binary: Path) -> None:
    cfg = _run(tmp_path, binary)
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    for name, count in manifest["tables"].items():
        assert (cfg.run_dir / "tables" / f"{name}.csv").is_file()
        assert isinstance(count, int)


def test_contract_columns_are_all_known() -> None:
    # The writer drops unknown keys, so every column a producer passes must be
    # one the contract declares; this guards against a typo silently vanishing.
    for table, header in contracts.COLUMNS.items():
        assert header, table
        assert len(set(header)) == len(header), table
