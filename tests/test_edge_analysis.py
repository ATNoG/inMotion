"""Interference statistics: bootstrap determinism, verdicts and both tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from inmotion.edge import analysis, contracts
from inmotion.edge.config import read_table, write_table


def _write_rows(path: Path, header: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, -1) for key in header})


def _stress_fixture(
    tmp_path: Path, *, device: str = "wave-router", reps: int = 3
) -> tuple[Path, Path]:
    """Canonical stress tables plus the raw sidecars analysis reads."""
    tables = tmp_path / "tables"
    raw = tmp_path / "raw" / "stress"
    raw.mkdir(parents=True)
    busy = {"baseline": 10.0, "inference": 12.0, "stress": 45.0, "both": 47.0}
    raw_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    interfaces: list[dict[str, object]] = []
    inference: list[dict[str, object]] = []

    for rep in range(reps):
        for phase, cpu in busy.items():
            window = f"w-{phase}-{rep}"
            for step in range(4):
                raw_rows.append(
                    {
                        "window": window,
                        "device": device,
                        "phase": phase,
                        "repetition": rep,
                        "t_rel_s": step * 0.5,
                        "cpu_busy_pct": cpu + (step % 2),
                        "cpu_iowait_pct": 0.5,
                        "cpu_softirq_pct": 0.1,
                        "cpu_sirq_pct": 0.2,
                        "per_core_busy_pct": "10.0;20.0",
                        "mem_available_kb": 1_800_000,
                        "loadavg_1m": 1.0 + rep * 0.1,
                        "rx_bytes": 1000 + step * 100,
                        "tx_bytes": 500 + step * 50,
                        "rx_drops": 0,
                        "tx_drops": 0,
                        "rx_errors": 0,
                        "tx_errors": 0,
                        "temp_c": 45.0,
                        "freq_mhz": 1500.0,
                    }
                )
            for probe in ("lan", "fwd"):
                lost = 1 if phase in ("stress", "both") and probe == "lan" else 0
                for seq in range(10):
                    probe_rows.append(
                        {
                            "window": window,
                            "device": device,
                            "phase": phase,
                            "repetition": rep,
                            "probe": probe,
                            "seq": seq,
                            "rtt_ms": -1.0 if seq == 9 and lost else 0.4 + seq * 0.01,
                            "lost": 1 if seq == 9 and lost else 0,
                        }
                    )
            interfaces.append(
                {
                    "window": window,
                    "device": device,
                    "phase": phase,
                    "repetition": rep,
                    "rx_packets": 8000,
                    "tx_packets": 2000,
                    "rx_drops": 5 if phase in ("stress", "both") else 0,
                    "tx_drops": 0,
                    "rx_errors": 0,
                    "tx_errors": 0,
                    "duration_s": 1.5,
                }
            )
        for phase in ("inference", "both"):
            inference.append(
                {
                    "window": f"w-{phase}-{rep}",
                    "device": device,
                    "phase": phase,
                    "repetition": rep,
                    "infer_p50_ms": 2.0 if phase == "inference" else 3.0,
                    "infer_p95_ms": 2.6 if phase == "inference" else 4.0,
                    "infer_cpu_ms_per_infer": 1.8 if phase == "inference" else 2.4,
                    "errors": 0,
                    "status": "ok",
                }
            )

    write_table(tables / "stress_raw.csv", "stress_raw", raw_rows)
    write_table(tables / "stress_probes.csv", "stress_probes", probe_rows)
    _write_rows(
        raw / "interfaces.csv",
        (
            "window",
            "device",
            "phase",
            "repetition",
            "rx_packets",
            "tx_packets",
            "rx_drops",
            "tx_drops",
            "rx_errors",
            "tx_errors",
            "duration_s",
        ),
        interfaces,
    )
    _write_rows(
        raw / "inference.csv",
        (
            "window",
            "device",
            "phase",
            "repetition",
            "infer_p50_ms",
            "infer_p95_ms",
            "infer_cpu_ms_per_infer",
            "errors",
            "status",
        ),
        inference,
    )
    return tables, tmp_path / "raw"


def test_block_bootstrap_is_deterministic_and_bounded() -> None:
    baseline = [10.0, 11.0, 9.0, 10.5, 9.5]
    stress = [40.0, 42.0, 39.0, 41.0, 40.5]
    first = analysis.block_bootstrap(baseline, stress)
    second = analysis.block_bootstrap(baseline, stress)
    assert first == second
    delta, lo, hi = first
    assert delta == pytest.approx(30.5)
    assert lo <= delta <= hi
    assert lo > 0


def test_block_bootstrap_needs_two_repetitions_for_a_ci() -> None:
    delta, lo, hi = analysis.block_bootstrap([1.0], [2.0])
    assert delta == 1.0
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_block_bootstrap_ignores_missing_pairs() -> None:
    delta, _lo, _hi = analysis.block_bootstrap([1.0, math.nan, 3.0], [2.0, 4.0, 5.0])
    assert delta == pytest.approx(1.5)


def test_wilcoxon_and_effect_size() -> None:
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
    increased = [2.0, 3.0, 4.0, 5.0, 6.0]
    assert 0.0 <= analysis.wilcoxon_p_value(baseline, increased) <= 1.0
    assert analysis.wilcoxon_p_value(baseline, baseline) == 1.0
    assert analysis.paired_effect_size(baseline, increased) > 0
    assert analysis.paired_effect_size(baseline, baseline) == 0.0
    assert analysis.paired_effect_size([1.0], [2.0]) == -1.0


def test_verdict_rules() -> None:
    assert analysis.verdict(1.0, 0.2, 2.0, 8.0, higher_is_worse=True) == "equivalent"
    assert analysis.verdict(10.0, 9.0, 12.0, 8.0, higher_is_worse=True) == "regression"
    assert analysis.verdict(-10.0, -12.0, -9.0, 8.0, higher_is_worse=True) == "improvement"
    assert analysis.verdict(0.0, -9.0, 9.0, 8.0, higher_is_worse=True) == "inconclusive"
    assert analysis.verdict(0.0, -9.0, 9.0, -1.0, higher_is_worse=True) == "inconclusive"
    # A KPI where lower is worse (available memory): a drop beyond the margin is
    # a regression.
    assert analysis.verdict(-500.0, -600.0, -400.0, 100.0, higher_is_worse=False) == "regression"


def test_run_emits_router_and_inference_kpis(tmp_path: Path) -> None:
    tables, raw = _stress_fixture(tmp_path)
    analysis.run(tables, raw)
    impact = {row["kpi"]: row for row in read_table(tables / "impact.csv", "impact")}
    assert impact["cpu_busy_pct"]["verdict"] == "regression"
    assert float(impact["cpu_busy_pct"]["delta_stress"]) == pytest.approx(35.0)
    assert impact["rtt_p95_ms_lan"]["unit"] == "ms"
    assert impact["rtt_p95_ms_fwd"]["unit"] == "ms"
    assert float(impact["drop_rate_pct"]["baseline_median"]) == 0.0
    assert float(impact["drop_rate_pct"]["stress_median"]) == pytest.approx(0.05)
    assert float(impact["rx_mbps"]["baseline_median"]) > 0
    for phase in contracts.STRESS_PHASES:
        assert f"{phase}_median" in impact["cpu_busy_pct"]

    inference = {
        row["kpi"]: row for row in read_table(tables / "inference_impact.csv", "inference_impact")
    }
    assert set(inference) == set(contracts.INFERENCE_KPIS)
    assert float(inference["infer_p95_ms"]["unloaded_median"]) == pytest.approx(2.6)
    assert float(inference["infer_p95_ms"]["loaded_median"]) == pytest.approx(4.0)
    assert float(inference["infer_p95_ms"]["margin"]) == pytest.approx(0.3 * 2.6)
    assert inference["infer_p95_ms"]["verdict"] == "regression"


def test_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    tables, raw = _stress_fixture(tmp_path)
    analysis.run(tables, raw)
    first = (tables / "impact.csv").read_bytes()
    analysis.run(tables, raw)
    assert (tables / "impact.csv").read_bytes() == first


def test_missing_stress_tables_write_headers_only(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    analysis.run(tables, None)
    assert (tables / "impact.csv").read_text().splitlines()[0] == ",".join(
        contracts.COLUMNS["impact"]
    )
    assert (tables / "inference_impact.csv").read_text().splitlines()[0] == ",".join(
        contracts.COLUMNS["inference_impact"]
    )


def test_analysis_never_puts_inference_kpis_in_impact(tmp_path: Path) -> None:
    tables, raw = _stress_fixture(tmp_path)
    analysis.run(tables, raw)
    impact_kpis = {row["kpi"] for row in read_table(tables / "impact.csv", "impact")}
    assert not impact_kpis & set(contracts.INFERENCE_KPIS)
    assert impact_kpis <= set(contracts.ROUTER_KPIS)
