"""Shared contract for the edge campaign: names, schemas, constants.

Every producer and consumer in ``inmotion.edge`` imports its column names,
precision sets, device identities, power assumptions and equivalence margins
from here, so a CSV header cannot drift from the writer that fills it or from
the figure that reads it. Nothing in this module performs I/O.

Canonical choices (see ``.dsh/plans/edge-contracts.md`` for the rationale):

* the five *registered* models, on ``dataset.csv`` with split seed 42;
* nine runtime-loadable precisions in the figures, three cast-only ones only in
  the supplementary table;
* effective bits include scales and zero points, so no scheme is drawn at a
  width it does not occupy.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Canonical model set and evaluation protocol ───────────────────────────

MODELS: tuple[str, ...] = ("lejepa", "t_jepa", "ts_jepa", "cf_jepa", "sigreg")

#: Precisions the Rust runtime can load from an artifact.
PRECISIONS_DEPLOYABLE: tuple[str, ...] = (
    "fp32",
    "int8",
    "q8",
    "q6",
    "q5",
    "q4",
    "q3",
    "q2",
    "q1",
)

#: Cast-only precisions: measured for accuracy, never plotted on the main axis
#: and never sent to a device, because the artifact reader has no F16/BF16.
PRECISIONS_CAST: tuple[str, ...] = ("bf16", "fp16", "fp8")

DATASET = "dataset.csv"
SEED = 42
N_TEST = 703

#: The test split must reproduce these fp32 MCCs bit-for-bit, or the run is
#: untrustworthy. Copied from ``inmotion.quant.evaluate.BASELINE_MCC`` on
#: purpose: the campaign must fail on its own if the harness moves, rather than
#: silently inheriting a changed baseline.
BASELINE_MCC: dict[str, float] = {
    "lejepa": 0.9142641066646362,
    "t_jepa": 0.8982207226632052,
    "ts_jepa": 0.9194244062588919,
    "cf_jepa": 0.9083210968710779,
    "sigreg": 0.6333303415465718,
}


# ── Devices ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeviceSpec:
    """One wavecom router, as the measurement harness addresses it."""

    key: str
    label: str
    host: str
    port: int
    user: str
    jump: str | None
    auth: str
    cpu: str
    cores: int
    ram_mb: int
    #: CPU package thermal design power, watts. Vendor figure, used only by the
    #: modelled power estimate and printed in the report next to every number.
    tdp_w: float


DEVICES: dict[str, DeviceSpec] = {
    "wave-router": DeviceSpec(
        key="wave-router",
        label="Device 1 · Q99W84AV9VIS",
        host="192.168.93.13",
        port=56022,
        user="admin",
        jump=None,
        auth="key",
        cpu="Intel Celeron J3455 @ 1.50 GHz",
        cores=4,
        ram_mb=3854,
        tdp_w=10.0,
    ),
    "wave-router-2": DeviceSpec(
        key="wave-router-2",
        label="Device 2 · PNLAFFQ9SJVW",
        host="10.33.0.113",
        port=56022,
        user="admin",
        jump="admin@192.168.93.13:56022",
        auth="password",
        cpu="AMD GX-412TC SOC @ 1.40 GHz",
        cores=4,
        ram_mb=1912,
        tdp_w=6.0,
    ),
}

#: Where the campaign deploys its binary and data on a router.
REMOTE_ROOT = "/mnt/extra/inmotion/edge"


# ── Power model ───────────────────────────────────────────────────────────

#: The SoCs expose no energy counter (no powercap, no hwmon energy/power, no
#: battery). The estimate below is a CPU-package model, not wall power:
#:
#:     u = cpu_seconds / (wall_seconds * cores)
#:     P(u) = P_idle + (P_TDP - P_idle) * clamp(u, 0, 1)
#:
#: ``P_idle`` is not measurable either, so every reported value carries the
#: range produced by these idle fractions of TDP. The measured inputs (CPU
#: seconds, wall seconds, cores) are stored beside the estimate so it can be
#: recomputed with different constants.
POWER_IDLE_FRACTIONS: tuple[float, ...] = (0.15, 0.30, 0.50)


# ── Interference study ────────────────────────────────────────────────────

#: Pre-declared margins: a change is declared equivalent only when the upper
#: bound of its 95 % CI is inside the margin. Chosen from the probe resolution
#: and the deployment's tolerance, not from the observed data.
EQUIVALENCE_MARGINS: dict[str, float] = {
    "cpu_busy_pct": 8.0,  # percentage points, device-wide
    "rtt_p95_ms_lan": 1.0,  # device-to-device LAN probe
    "rtt_p95_ms_fwd": 2.0,  # forwarded path through the router under test
    "drop_rate_pct": 0.01,  # packets dropped / packets seen
    "infer_p95_rel": 0.30,  # inference p95 under stress vs baseline
}

STRESS_PHASES: tuple[str, ...] = (
    "baseline",
    "inference",
    "stress",
    "both",
)

#: Router-health KPIs. ``impact.csv`` carries one row per KPI: the four phase
#: medians plus the inference-vs-baseline and stress-vs-baseline CIs.
ROUTER_KPIS: tuple[str, ...] = (
    "cpu_busy_pct",
    "cpu_iowait_pct",
    "mem_available_kb",
    "loadavg_1m",
    "rtt_p50_ms_lan",
    "rtt_p95_ms_lan",
    "rtt_p99_ms_lan",
    "rtt_loss_pct",
    "rtt_p95_ms_fwd",
    "drop_rate_pct",
    "rx_mbps",
    "tx_mbps",
)

#: Inference KPIs. These have no meaning in the baseline/stress phases (no
#: inference is running there), so they get their own table with an explicit
#: unloaded (inference-only) and loaded (both) pair instead of being forced into
#: the four-phase layout above.
INFERENCE_KPIS: tuple[str, ...] = (
    "infer_p50_ms",
    "infer_p95_ms",
    "infer_cpu_ms_per_infer",
)

#: ``artifact_sha256`` is the SHA-256 of ``model.safetensors`` only, everywhere.
#: One hashable file on the routers; the scheme lives in ``meta.json`` and the
#: weights file already encodes the quantized codes and scales.
ARTIFACT_HASH_FILE = "model.safetensors"


# ── CSV schemas ───────────────────────────────────────────────────────────
#
# One tuple per table. Writers build the header from the tuple; readers assert
# the header they find. ``-1`` means "this device cannot supply the value", the
# repo's existing convention for a numeric column (never an empty cell).

COLUMNS: dict[str, tuple[str, ...]] = {
    "artifacts": (
        "model",
        "precision",
        "runtime_loadable",
        "path",
        "bytes",
        "sha256",
        "status",
        "error",
    ),
    "accuracy": (
        "model",
        "checkpoint_sha256",
        "precision",
        "runtime_loadable",
        "n_params",
        "artifact_bytes",
        "effective_bits",
        "mcc",
        "mcc_lo",
        "mcc_hi",
        "accuracy",
        "accuracy_lo",
        "accuracy_hi",
        "f1_macro",
        "f1_lo",
        "f1_hi",
        "delta_mcc",
        "delta_lo",
        "delta_hi",
        "max_weight_error",
        "n_test",
        "dataset",
        "seed",
        "artifact_sha256",
    ),
    "per_class": (
        "model",
        "precision",
        "class",
        "support",
        "precision_c",
        "recall_c",
        "f1_c",
        "tp",
        "fp",
        "fn",
    ),
    "confusion": (
        "model",
        "precision",
        "true_class",
        "predicted_class",
        "count",
    ),
    "replay": (
        "device",
        "model",
        "precision",
        "samples",
        "agreement",
        "mcc_python",
        "mcc_device",
        "mcc_delta",
        "max_logit_diff",
        "parity_ok",
        "artifact_sha256",
        "status",
    ),
    "load": (
        "device",
        "model",
        "precision",
        "rep",
        "cold",
        "load_ms",
        "cpu_ms_load",
        "rss_after_load_kb",
        "artifact_bytes",
        "governor",
        "freq_mhz",
        "status",
    ),
    "bench": (
        "device",
        "model",
        "precision",
        "rep",
        "cpus",
        "concurrency",
        "iterations",
        "min_ms",
        "p50_ms",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "throughput_s",
        "cpu_ms_bench",
        "cpu_pct_bench",
        "cpu_ms_per_infer",
        "peak_rss_kb",
        "artifact_bytes",
        "energy_mj_est",
        "energy_mj_lo",
        "energy_mj_hi",
        "energy_mj_meas",
        "power_source",
        "freq_mhz",
        "temp_c_before",
        "temp_c_after",
        "foreign_cpu_pct",
        "loadavg_before",
        "loadavg_after",
        "binary_sha256",
        "artifact_sha256",
        "status",
    ),
    "cores": (
        "device",
        "model",
        "precision",
        "rep",
        "cpus",
        "cores",
        "concurrency",
        "p50_ms",
        "p95_ms",
        "throughput_s",
        "cpu_ms_per_infer",
        "foreign_cpu_pct",
        "status",
    ),
    "stress_raw": (
        "window",
        "device",
        "phase",
        "repetition",
        "t_rel_s",
        "cpu_busy_pct",
        "cpu_iowait_pct",
        "cpu_softirq_pct",
        "cpu_sirq_pct",
        "per_core_busy_pct",
        "mem_available_kb",
        "loadavg_1m",
        "rx_bytes",
        "tx_bytes",
        "rx_drops",
        "tx_drops",
        "rx_errors",
        "tx_errors",
        "temp_c",
        "freq_mhz",
    ),
    "stress_probes": (
        "window",
        "device",
        "phase",
        "repetition",
        "probe",
        "seq",
        "rtt_ms",
        "lost",
    ),
    "impact": (
        "device",
        "kpi",
        "unit",
        "baseline_median",
        "inference_median",
        "stress_median",
        "both_median",
        "delta_inference",
        "delta_inference_lo",
        "delta_inference_hi",
        "delta_stress",
        "delta_stress_lo",
        "delta_stress_hi",
        "p_value",
        "effect_size",
        "margin",
        "verdict",
    ),
    "inference_impact": (
        "device",
        "kpi",
        "unit",
        "unloaded_median",
        "loaded_median",
        "delta",
        "delta_lo",
        "delta_hi",
        "margin",
        "verdict",
    ),
    "power": (
        "device",
        "model",
        "precision",
        "cores",
        "wall_s",
        "cpu_s",
        "utilization",
        "mj_per_infer_est",
        "mj_per_infer_lo",
        "mj_per_infer_hi",
        "mj_per_infer_meas",
        "source",
        "assumptions",
    ),
}


# ── Figures ───────────────────────────────────────────────────────────────

FIGURE_STEMS: tuple[str, ...] = (
    "fig1_accuracy_vs_bits",
    "fig2_failure_modes",
    "fig3_size_accuracy",
    "fig4_device_accuracy",
    "fig5_runtime_cost",
    "fig6_core_scaling",
    "fig7_router_impact",
    "fig8_stress_robustness",
)


def precision_rank(precision: str) -> int:
    """Ordering index for a precision, best first; unknown precisions last."""
    order = PRECISIONS_DEPLOYABLE + PRECISIONS_CAST
    try:
        return order.index(precision)
    except ValueError:
        return len(order)


def effective_bits(artifact_bytes: int, n_params: int) -> float:
    """Bits per parameter actually stored, scales and metadata included."""
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    return artifact_bytes * 8.0 / n_params
