"""The eight edge figures must build from tables alone, reproducibly.

Every table a figure reads is synthesised here with the exact ``contracts``
schema, so the tests exercise the plotting path without a device, and the
reproducibility check runs the whole build twice and compares bytes. The
accuracy-side tables used by F1–F3 are also covered from the real writer in
``tests/test_edge_accuracy.py``; this file keeps the figures honest about their
inputs, not about the measurements.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from inmotion.edge import contracts as c  # noqa: E402
from inmotion.edge import figures  # noqa: E402

MODELS = list(c.MODELS)
DEVICES = list(c.DEVICES)
CLASSES = ["AA", "AB", "BA", "BB"]
N = 703

#: Rough artifact footprint per model at fp32, measured from the real export;
#: only the *shape* of the curve matters to the figures.
FP32_BYTES = {
    "lejepa": 2_284_698,
    "t_jepa": 2_554_490,
    "ts_jepa": 8_633_203,
    "cf_jepa": 4_944_607,
    "sigreg": 2_811_705,
}
N_PARAMS = {
    "lejepa": 565_765,
    "t_jepa": 633_477,
    "ts_jepa": 2_153_349,
    "cf_jepa": 1_232_261,
    "sigreg": 696_900,
}
SIZE_FACTOR = {
    "fp32": 1.0,
    "int8": 0.29,
    "q8": 0.30,
    "q6": 0.24,
    "q5": 0.21,
    "q4": 0.21,
    "q3": 0.16,
    "q2": 0.16,
    "q1": 0.09,
}
PENALTY = {
    "fp32": 0.0,
    "int8": 0.002,
    "q8": 0.004,
    "q6": 0.008,
    "q5": 0.012,
    "q4": 0.006,
    "q3": 0.02,
    "q2": 0.05,
    "q1": 0.30,
}


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _deployable_accuracy() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for precision in c.PRECISIONS_DEPLOYABLE:
            mcc = max(c.BASELINE_MCC[model] - PENALTY[precision], -0.05)
            artifact_bytes = int(FP32_BYTES[model] * SIZE_FACTOR[precision])
            rows.append(
                {
                    "model": model,
                    "checkpoint_sha256": _sha(model),
                    "precision": precision,
                    "runtime_loadable": "true",
                    "n_params": N_PARAMS[model],
                    "artifact_bytes": artifact_bytes,
                    "effective_bits": c.effective_bits(artifact_bytes, N_PARAMS[model]),
                    "mcc": mcc,
                    "mcc_lo": mcc - 0.025,
                    "mcc_hi": mcc + 0.020,
                    "accuracy": 0.9 - PENALTY[precision],
                    "accuracy_lo": 0.88 - PENALTY[precision],
                    "accuracy_hi": 0.92 - PENALTY[precision],
                    "f1_macro": 0.89 - PENALTY[precision],
                    "f1_lo": 0.87 - PENALTY[precision],
                    "f1_hi": 0.91 - PENALTY[precision],
                    "delta_mcc": -PENALTY[precision],
                    "delta_lo": -PENALTY[precision] - 0.01,
                    "delta_hi": -PENALTY[precision] + 0.005,
                    "max_weight_error": 0.01 + PENALTY[precision],
                    "n_test": N,
                    "dataset": c.DATASET,
                    "seed": c.SEED,
                    "artifact_sha256": _sha(f"{model}-{precision}"),
                }
            )
    return rows


def _cast_accuracy() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for precision in c.PRECISIONS_CAST:
            mcc = c.BASELINE_MCC[model] - (0.06 if precision == "fp8" else 0.0)
            rows.append(
                {
                    "model": model,
                    "checkpoint_sha256": _sha(model),
                    "precision": precision,
                    "runtime_loadable": "false",
                    "n_params": N_PARAMS[model],
                    "artifact_bytes": -1,
                    "effective_bits": 8.0 if precision == "fp8" else 16.0,
                    "mcc": mcc,
                    "mcc_lo": mcc - 0.025,
                    "mcc_hi": mcc + 0.020,
                    "accuracy": 0.9,
                    "accuracy_lo": 0.88,
                    "accuracy_hi": 0.92,
                    "f1_macro": 0.89,
                    "f1_lo": 0.87,
                    "f1_hi": 0.91,
                    "delta_mcc": mcc - c.BASELINE_MCC[model],
                    "delta_lo": -0.07,
                    "delta_hi": 0.0,
                    "max_weight_error": 0.02,
                    "n_test": N,
                    "dataset": c.DATASET,
                    "seed": c.SEED,
                    "artifact_sha256": "",
                }
            )
    return rows


def _confusion_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for precision in c.PRECISIONS_DEPLOYABLE:
            confidence = float(
                np.clip(0.25 + c.BASELINE_MCC[model] - PENALTY[precision], 0.25, 0.97)
            )
            matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
            per_class = N // len(CLASSES)
            for true_index in range(len(CLASSES)):
                diagonal = int(round(confidence * per_class))
                matrix[true_index, true_index] = diagonal
                residual = per_class - diagonal
                for offset in range(1, len(CLASSES)):
                    target = (true_index + offset) % len(CLASSES)
                    share = residual // (len(CLASSES) - offset)
                    matrix[true_index, target] += share
                matrix[true_index, true_index] += per_class - matrix[true_index].sum()
            # per_class uses 703 // 4; put the remainder on one diagonal cell so
            # every synthetic table sums to the contract's n_test.
            matrix[0, 0] += N - matrix.sum()
            for true_index, true_class in enumerate(CLASSES):
                for predicted_index, predicted_class in enumerate(CLASSES):
                    rows.append(
                        {
                            "model": model,
                            "precision": precision,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "count": int(matrix[true_index, predicted_index]),
                        }
                    )
    return rows


def _replay_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in DEVICES:
        for model in MODELS[:3]:
            for precision in ("fp32", "q4"):
                mcc = c.BASELINE_MCC[model] - PENALTY[precision]
                offset = 0.0004 * DEVICES.index(device)
                rows.append(
                    {
                        "device": device,
                        "model": model,
                        "precision": precision,
                        "samples": 703 if device == "wave-router" else 200,
                        "agreement": 0.9955 - 0.0004 * DEVICES.index(device),
                        "mcc_python": mcc,
                        "mcc_device": mcc + 0.001 + offset,
                        "mcc_delta": 0.001 + offset,
                        "max_logit_diff": 1e-6,
                        "parity_ok": "true",
                        "artifact_sha256": _sha(f"{model}-{precision}"),
                        "status": "ok",
                    }
                )
    return rows


def _bench_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device_index, device in enumerate(DEVICES):
        for model_index, model in enumerate(MODELS):
            for precision_index, precision in enumerate(c.PRECISIONS_DEPLOYABLE):
                for rep in range(3):
                    p50 = (1.2 + 0.15 * precision_index + 0.2 * model_index + 0.01 * rep) * (
                        1.0 + 0.22 * device_index
                    )
                    rows.append(
                        {
                            "device": device,
                            "model": model,
                            "precision": precision,
                            "rep": rep,
                            "cpus": "0-3",
                            "concurrency": 1,
                            "iterations": 120,
                            "min_ms": p50 * 0.8,
                            "p50_ms": p50,
                            "mean_ms": p50 * 1.05,
                            "p95_ms": p50 * 1.5,
                            "p99_ms": p50 * 1.8,
                            "max_ms": p50 * 2.2,
                            "throughput_s": 1000.0 / p50,
                            "cpu_ms_bench": p50 * 120,
                            "cpu_pct_bench": 92.0,
                            "cpu_ms_per_infer": p50 * 0.9,
                            "peak_rss_kb": 24_000 + 4_000 * model_index + 2_500 * device_index,
                            "artifact_bytes": int(FP32_BYTES[model] * SIZE_FACTOR[precision]),
                            "energy_mj_est": 0.05 + 0.002 * precision_index,
                            "energy_mj_lo": 0.04 + 0.002 * precision_index,
                            "energy_mj_hi": 0.07 + 0.002 * precision_index,
                            "energy_mj_meas": -1,
                            "power_source": "model",
                            "freq_mhz": 1500,
                            "temp_c_before": 45.0,
                            "temp_c_after": 47.0,
                            "foreign_cpu_pct": 1.5,
                            "loadavg_before": 0.2,
                            "loadavg_after": 0.3,
                            "binary_sha256": _sha("binary"),
                            "artifact_sha256": _sha(f"{model}-{precision}"),
                            "status": "ok",
                        }
                    )
    return rows


def _load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in DEVICES:
        for model in MODELS:
            for precision in ("fp32", "q4", "q2"):
                for rep in range(2):
                    for cold in ("warm", "cold"):
                        rows.append(
                            {
                                "device": device,
                                "model": model,
                                "precision": precision,
                                "rep": rep,
                                "cold": "true" if cold == "cold" else "false",
                                "load_ms": (
                                    (10.0 + 2.0 * DEVICES.index(device) + 1.0 * MODELS.index(model))
                                    if cold == "warm"
                                    else 40.0
                                ),
                                "cpu_ms_load": 10.0,
                                "rss_after_load_kb": 22_000,
                                "artifact_bytes": int(FP32_BYTES[model] * SIZE_FACTOR[precision]),
                                "governor": "powersave",
                                "freq_mhz": 1500,
                                "status": "ok",
                            }
                        )
    return rows


def _power_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in DEVICES:
        for model in MODELS:
            for precision in c.PRECISIONS_DEPLOYABLE:
                energy = 0.06 * (1.0 + 0.3 * DEVICES.index(device))
                rows.append(
                    {
                        "device": device,
                        "model": model,
                        "precision": precision,
                        "cores": 4,
                        "wall_s": 2.1,
                        "cpu_s": 1.9,
                        "utilization": 0.9,
                        "mj_per_infer_est": energy,
                        "mj_per_infer_lo": energy * 0.7,
                        "mj_per_infer_hi": energy * 1.3,
                        "mj_per_infer_meas": -1,
                        "source": "model",
                        "assumptions": "P_idle=0.3*TDP",
                    }
                )
    return rows


def _cores_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS[:3]:
        for rep in range(2):
            for cores in (1, 2, 3, 4):
                rows.append(
                    {
                        "device": DEVICES[0],
                        "model": model,
                        "precision": "q4",
                        "rep": rep,
                        "cpus": f"0-{cores - 1}",
                        "cores": cores,
                        "concurrency": 1,
                        "p50_ms": 4.0 / cores * (1.0 + 0.15 * MODELS.index(model)),
                        "p95_ms": 6.0 / cores * (1.0 + 0.15 * MODELS.index(model)),
                        "throughput_s": (200.0 * cores / (1.0 + 0.15 * MODELS.index(model))),
                        "cpu_ms_per_infer": 4.0,
                        "foreign_cpu_pct": 1.0,
                        "status": "ok",
                    }
                )
            for concurrency in (2, 4, 8):
                rows.append(
                    {
                        "device": DEVICES[0],
                        "model": model,
                        "precision": "q4",
                        "rep": rep,
                        "cpus": "0-3",
                        "cores": 4,
                        "concurrency": concurrency,
                        "p50_ms": 5.0,
                        "p95_ms": 7.0,
                        "throughput_s": (
                            200.0 * 4 / (1.0 + 0.15 * MODELS.index(model)) * concurrency * 0.9
                        ),
                        "cpu_ms_per_infer": 4.0,
                        "foreign_cpu_pct": 1.0,
                        "status": "ok",
                    }
                )
    return rows


def _impact_rows(
    kpis: tuple[tuple[str, str, float, float, float, float, float, float, float, float], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in DEVICES:
        for kpi, unit, margin, base, inference, stress, both, d_lo, d_hi, p_value in kpis:
            rows.append(
                {
                    "device": device,
                    "kpi": kpi,
                    "unit": unit,
                    "baseline_median": base,
                    "inference_median": inference,
                    "stress_median": stress,
                    "both_median": both,
                    "delta_inference": inference - base,
                    "delta_inference_lo": d_lo,
                    "delta_inference_hi": d_hi,
                    "delta_stress": stress - base,
                    "delta_stress_lo": stress - base - abs(d_hi),
                    "delta_stress_hi": stress - base + abs(d_hi),
                    "p_value": p_value,
                    "effect_size": 0.8,
                    "margin": margin,
                    "verdict": "equivalent",
                }
            )
    return rows


ROUTER_ROWS = (
    ("cpu_busy_pct", "%", 8.0, 3.0, 5.0, 85.0, 88.0, 1.0, 3.0, 0.01),
    ("rtt_p95_ms_lan", "ms", 1.0, 0.40, 0.45, 1.20, 1.25, -0.02, 0.10, 0.04),
    ("drop_rate_pct", "%", 0.01, 0.0, 0.0, 0.02, 0.03, 0.0, 0.0, 0.5),
)

INFERENCE_ROWS = (
    ("infer_p50_ms", "ms", 2.0, 2.3, 0.30, 0.2, 0.35),
    ("infer_p95_ms", "ms", 3.0, 3.6, 0.60, 0.4, 0.75),
    ("infer_cpu_ms_per_infer", "ms", 1.8, 2.1, 0.30, 0.2, 0.4),
)


def _inference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in DEVICES:
        for kpi, unit, unloaded, loaded, margin, delta_lo, delta_hi in INFERENCE_ROWS:
            rows.append(
                {
                    "device": device,
                    "kpi": kpi,
                    "unit": unit,
                    "unloaded_median": unloaded,
                    "loaded_median": loaded,
                    "delta": loaded - unloaded,
                    "delta_lo": delta_lo,
                    "delta_hi": delta_hi,
                    "margin": margin,
                    "verdict": "equivalent",
                }
            )
    return rows


def _write(directory: Path, table: str, rows: list[dict[str, object]]) -> Path:
    columns = list(c.COLUMNS[table])
    frame = pd.DataFrame(rows, columns=columns)
    path = directory / f"{table}.csv"
    frame.to_csv(path, index=False)
    return path


def _synthetic_tables(directory: Path) -> Path:
    """Every table the figures read, with the exact shared schema."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory, "accuracy", _deployable_accuracy() + _cast_accuracy())
    _write(directory, "confusion", _confusion_rows())
    _write(directory, "replay", _replay_rows())
    _write(directory, "bench", _bench_rows())
    _write(directory, "load", _load_rows())
    _write(directory, "power", _power_rows())
    _write(directory, "cores", _cores_rows())
    _write(directory, "impact", _impact_rows(ROUTER_ROWS))
    _write(directory, "inference_impact", _inference_rows())
    return directory


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Style contract ───────────────────────────────────────────────────────


def test_style_is_serif_8pt_and_embeds_fonts() -> None:
    figures._style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert plt.rcParams["font.size"] == pytest.approx(8.0)
    family = plt.rcParams["font.family"]
    assert "serif" in (family if isinstance(family, list) else [family])
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["axes.spines.top"] is False


# ── Full build and reproducibility ───────────────────────────────────────


def test_build_writes_eight_figures_data_and_is_byte_identical(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    out = tmp_path / "figures"

    first = figures.build(tables, out)
    assert len(first) == len(c.FIGURE_STEMS) * 3  # data csv + pdf + png per stem
    for stem in c.FIGURE_STEMS:
        data = out / "data" / f"{stem}.csv"
        pdf = out / f"{stem}.pdf"
        png = out / f"{stem}.png"
        for path in (data, pdf, png):
            assert path.is_file(), path
        assert data.stat().st_size > 0
        assert pdf.stat().st_size > 5_000
        assert png.stat().st_size > 5_000
        frame = pd.read_csv(data, float_precision="round_trip")
        assert len(frame) > 0
        assert not frame.isna().all(axis=None)

    before = {path: _digest(path) for path in first}
    second = figures.build(tables, out)
    after = {path: _digest(path) for path in second}
    assert before == after
    assert set(before) == set(after)


def test_build_refuses_unknown_stem(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    with pytest.raises(ValueError, match="unknown figure stem"):
        figures.build(tables, tmp_path / "figures", only=("fig9_imaginary",))


# ── Missing input failures name the table and the command ───────────────


@pytest.mark.parametrize(
    ("stem", "table", "command"),
    [
        ("fig4_device_accuracy", "replay", "inmotion edge replay"),
        ("fig5_runtime_cost", "power", "inmotion edge power"),
        ("fig6_core_scaling", "cores", "inmotion edge bench"),
        ("fig7_router_impact", "impact", "inmotion edge analyze"),
        ("fig8_stress_robustness", "inference_impact", "inmotion edge analyze"),
    ],
)
def test_missing_table_error_names_table_and_command(
    tmp_path: Path, stem: str, table: str, command: str
) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    (tables / f"{table}.csv").unlink()
    with pytest.raises(FileNotFoundError) as caught:
        figures.build(tables, tmp_path / "figures", only=(stem,))
    message = str(caught.value)
    assert f"{table}.csv" in message
    assert command in message


def test_missing_router_kpi_error_names_kpi_and_table(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    impact = pd.read_csv(tables / "impact.csv")
    impact = impact[impact["kpi"] != "cpu_busy_pct"]
    impact.to_csv(tables / "impact.csv", index=False)
    with pytest.raises(ValueError, match="cpu_busy_pct") as caught:
        figures.build(tables, tmp_path / "figures", only=("fig7_router_impact",))
    message = str(caught.value)
    assert "cpu_busy_pct" in message
    assert "impact.csv" in message


def test_missing_inference_kpi_error_names_kpi(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    inference = pd.read_csv(tables / "inference_impact.csv")
    inference = inference[inference["kpi"] != "infer_p95_ms"]
    inference.to_csv(tables / "inference_impact.csv", index=False)
    with pytest.raises(ValueError, match="infer_p95_ms") as caught:
        figures.build(tables, tmp_path / "figures", only=("fig8_stress_robustness",))
    assert "infer_p95_ms" in str(caught.value)


# ── Per-figure data contracts ───────────────────────────────────────────


def test_fig1_plots_deployable_rows_only_with_paired_cis(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig1_accuracy_vs_bits",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig1_accuracy_vs_bits.csv")
    assert len(frame) == len(c.MODELS) * len(c.PRECISIONS_DEPLOYABLE)
    assert set(frame["precision"]) == set(c.PRECISIONS_DEPLOYABLE)
    assert not set(frame["precision"]) & set(c.PRECISIONS_CAST)
    for column in ("effective_bits", "mcc_lo", "mcc_hi", "delta_lo", "delta_hi"):
        assert column in frame.columns
    fp32 = frame[frame["precision"] == "fp32"]
    assert fp32["delta_mcc"].eq(0.0).all() or fp32["delta_mcc"].abs().max() < 1e-12


def test_fig2_data_contains_confusion_and_majority_rows(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig2_failure_modes",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig2_failure_modes.csv")
    assert set(frame["panel"]) == {"confusion", "majority"}
    confusion = frame[frame["panel"] == "confusion"]
    assert set(confusion["precision"]) == {"fp32", "q4", "q2"}
    assert confusion["count"].sum() == 3 * N
    majority = frame[frame["panel"] == "majority"]
    assert majority["share"].between(0.0, 1.0).all()


def test_fig4_data_joins_python_ci(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig4_device_accuracy",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig4_device_accuracy.csv")
    assert set(frame["device"]) == set(c.DEVICES)
    assert (frame["mcc_python_lo"] <= frame["mcc_python"]).all()
    assert (frame["mcc_python"] <= frame["mcc_python_hi"]).all()


def test_fig5_data_has_all_six_panels(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig5_runtime_cost",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig5_runtime_cost.csv")
    assert len(set(frame["panel"])) == 6
    assert (frame["lo"] <= frame["value"]).all()
    assert (frame["value"] <= frame["hi"]).all()


def test_fig7_falls_back_to_forwarded_rtt(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    impact = pd.read_csv(tables / "impact.csv")
    lan = impact[impact["kpi"] == "rtt_p95_ms_lan"].copy()
    forwarded = lan.assign(kpi="rtt_p95_ms_fwd")
    impact = impact[impact["kpi"] != "rtt_p95_ms_lan"]
    pd.concat([impact, forwarded], ignore_index=True).to_csv(tables / "impact.csv", index=False)
    figures.build(tables, tmp_path / "figures", only=("fig7_router_impact",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig7_router_impact.csv")
    assert "rtt_p95_ms_fwd" in set(frame["kpi"])
    # 'both' has no CI in impact.csv and must be drawn as median-only.
    both = frame[frame["phase"] == "both"]
    assert not both["has_ci"].any() if both["has_ci"].dtype == bool else True


def test_fig8_uses_inference_impact_and_marks_ci(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig8_stress_robustness",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig8_stress_robustness.csv")
    assert set(frame["kpi"]) == set(c.INFERENCE_KPIS)
    assert (frame["ci_low_abs"] <= frame["ci_high_abs"]).all()
    assert (frame["margin"] > 0).all()


def test_data_csvs_quote_effective_bits_not_nominal_width(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig1_accuracy_vs_bits",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig1_accuracy_vs_bits.csv")
    fp32 = frame[frame["precision"] == "fp32"].iloc[0]
    lejepa = frame[(frame["model"] == "lejepa") & (frame["precision"] == "fp32")].iloc[0]
    expected = c.effective_bits(FP32_BYTES["lejepa"], N_PARAMS["lejepa"])
    assert fp32["model"] != ""
    assert lejepa["effective_bits"] == pytest.approx(expected)
    assert lejepa["effective_bits"] > 32.0  # metadata is counted, so never nominal


def test_pareto_buckets_storage_ties_and_keeps_best(tmp_path: Path) -> None:
    # Sizes come from the canonical lejepa/ts_jepa rows: q1 is 34 bytes smaller
    # than q3, q2/q4/q5 share almost the same byte count. Only the best accuracy
    # inside a tier may survive, or a collapsed scheme would win on 34 bytes.
    frame = pd.DataFrame(
        {
            "size_mb": [0.412183, 0.412217, 0.481965, 0.481993, 0.482001, 0.691329, 2.284243],
            "mcc": [-0.0013, 0.8377, 0.3231, 0.9083, 0.9085, 0.9143, 0.9194],
        }
    )
    front = figures._pareto(frame, x="size_mb", y="mcc")
    assert front == {1, 4, 5, 6}


def test_fig3_pareto_front_is_truly_non_dominated(tmp_path: Path) -> None:
    tables = _synthetic_tables(tmp_path / "tables")
    figures.build(tables, tmp_path / "figures", only=("fig3_size_accuracy",))
    frame = pd.read_csv(tmp_path / "figures" / "data" / "fig3_size_accuracy.csv")
    points = frame[["size_mb", "mcc"]].to_numpy()
    on_front = frame["on_pareto"].astype(bool).to_numpy()

    def dominates(j: int, i: int) -> bool:
        weaker_size = points[j, 0] <= points[i, 0]
        better_score = points[j, 1] >= points[i, 1]
        strict = points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]
        return bool(weaker_size and better_score and strict)

    for i in np.where(on_front)[0]:
        offenders = [j for j in range(len(frame)) if j != i and dominates(j, i)]
        assert not offenders, (frame.iloc[i].to_dict(), frame.iloc[offenders].to_dict())

    for i in np.where(~on_front)[0]:
        explained = any(dominates(j, i) for j in np.where(on_front)[0])
        if not explained:
            # Only a storage-tier tie may exclude a point without strict domination.
            same_tier = np.abs(points[:, 0] - points[i, 0]) <= 1e-3 * points[i, 0]
            explained = any(
                same_tier[j] and on_front[j] and points[j, 1] >= points[i, 1]
                for j in range(len(frame))
            )
        assert explained, frame.iloc[i].to_dict()
