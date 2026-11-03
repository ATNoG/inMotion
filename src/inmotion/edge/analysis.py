"""Interference statistics: paired block bootstrap, Wilcoxon, verdicts.

The repetitions are the unit of resampling. Every window measured one phase
once, so a repetition gives one paired (baseline, phase) observation; drawing
whole repetitions with replacement keeps the phase pairing intact, which a
sample-level bootstrap would destroy.

Two tables come out of here:

``impact.csv``            the router-health KPIs from ``contracts.ROUTER_KPIS``,
                          four phase medians plus inference-vs-baseline and
                          stress-vs-baseline CIs, one row per KPI;
``inference_impact.csv``  the runtime's own latency under stress
                          (``contracts.INFERENCE_KPIS``), unloaded (inference-only
                          phase) against loaded (both phase).

A KPI is only written when there is data for it; ``-1`` inside a row means that
particular statistic is unavailable, and a row is omitted entirely when the
measurement does not exist (no forwarded-path probe, no interface counters).
No value is ever fabricated from a missing phase.

Verdicts use the pre-declared margins in ``contracts.EQUIVALENCE_MARGINS``. A
KPI without a declared margin gets ``margin=-1`` and ``inconclusive``: the
campaign reports the estimate, but no threshold was declared, so it cannot
claim equivalence and says so instead of inventing a margin.
"""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from inmotion.edge import contracts
from inmotion.edge.config import as_float, read_table, write_table

#: 5 000 draws, seed 42: the campaign's declared resampling budget.
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 42
CI = 95.0

#: KPIs where a larger value is worse. The rest of the measured KPIs are
#: "larger is better"; DIRECTION covers both and anything unlisted is treated as
#: informational (no verdict) by :func:`verdict`.
HIGHER_IS_WORSE = frozenset(
    {
        "cpu_busy_pct",
        "cpu_iowait_pct",
        "loadavg_1m",
        "rtt_p50_ms_lan",
        "rtt_p95_ms_lan",
        "rtt_p99_ms_lan",
        "rtt_loss_pct",
        "rtt_p95_ms_fwd",
        "drop_rate_pct",
    }
)
LOWER_IS_WORSE = frozenset({"mem_available_kb", "rx_mbps", "tx_mbps"})

#: Relative margins for the inference KPIs that have no pre-declared absolute
#: one. The p95 number is the contract's; the other two use the same 30 % of the
#: unloaded median so one table does not mix tolerances without saying so.
INFERENCE_REL_MARGINS: dict[str, float] = {
    "infer_p50_ms": 0.30,
    "infer_p95_ms": contracts.EQUIVALENCE_MARGINS["infer_p95_rel"],
    "infer_cpu_ms_per_infer": 0.30,
}

UNITS: dict[str, str] = {
    "cpu_busy_pct": "%",
    "cpu_iowait_pct": "%",
    "mem_available_kb": "kB",
    "loadavg_1m": "count",
    "rtt_p50_ms_lan": "ms",
    "rtt_p95_ms_lan": "ms",
    "rtt_p99_ms_lan": "ms",
    "rtt_loss_pct": "%",
    "rtt_p95_ms_fwd": "ms",
    "drop_rate_pct": "%",
    "rx_mbps": "Mbit/s",
    "tx_mbps": "Mbit/s",
    "infer_p50_ms": "ms",
    "infer_p95_ms": "ms",
    "infer_cpu_ms_per_infer": "ms",
}


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ── statistics ───────────────────────────────────────────────────────────


def block_bootstrap(
    baseline: Sequence[float],
    phase: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Paired block bootstrap over repetitions: median delta and its 95 % CI.

    ``baseline`` and ``phase`` are aligned by repetition. A draw takes whole
    repetition indices and applies them to both series, so the pairing that the
    experiment was designed around survives the resampling.

    With fewer than two repetitions there is nothing to resample: the point
    delta is returned with ``-1`` bounds, and the caller marks the KPI as
    inconclusive instead of inventing a spread.
    """
    pairs = [(float(a), float(b)) for a, b in zip(baseline, phase, strict=False)]
    pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        return math.nan, math.nan, math.nan
    point = statistics.median([b for _a, b in pairs]) - statistics.median([a for a, _b in pairs])
    if len(pairs) < 2:
        # Not enough repetitions to resample. NaN marks "no interval"; the CSV
        # writer renders it as the contract's -1, while verdict() can tell the
        # difference from a legitimate negative delta.
        return point, math.nan, math.nan
    base = np.asarray([a for a, _b in pairs], dtype=float)
    loaded = np.asarray([b for _a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(draws, len(pairs)))
    deltas = np.median(loaded[indices], axis=1) - np.median(base[indices], axis=1)
    lo = float(np.percentile(deltas, (100.0 - CI) / 2.0))
    hi = float(np.percentile(deltas, 100.0 - (100.0 - CI) / 2.0))
    return float(point), lo, hi


def wilcoxon_p_value(baseline: Sequence[float], phase: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the paired repetitions."""
    pairs = [
        (float(a), float(b))
        for a, b in zip(baseline, phase, strict=False)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 2:
        return -1.0
    differences = [b - a for a, b in pairs]
    if all(difference == 0 for difference in differences):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon([b for _a, b in pairs], [a for a, _b in pairs]).pvalue)
    except (ValueError, IndexError):
        return -1.0


def paired_effect_size(baseline: Sequence[float], phase: Sequence[float]) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff).

    A standardised mean change, always reported next to the raw difference
    because ``p`` alone cannot tell a large reliable shift from a tiny noisy one.
    """
    pairs = [
        (float(a), float(b))
        for a, b in zip(baseline, phase, strict=False)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 2:
        return -1.0
    differences = [b - a for a, b in pairs]
    spread = statistics.stdev(differences)
    if spread == 0:
        return (
            0.0 if statistics.fmean(differences) == 0 else math.copysign(math.inf, differences[0])
        )
    return statistics.fmean(differences) / spread


def verdict(delta: float, lo: float, hi: float, margin: float, *, higher_is_worse: bool) -> str:
    """Equivalence verdict from the CI against a pre-declared margin.

    ``equivalent``  the whole 95 % CI sits inside ±margin;
    ``regression``  the CI is entirely beyond the margin in the bad direction;
    ``improvement`` the CI is entirely beyond the margin in the good direction;
    ``inconclusive`` anything else, including an unavailable CI.
    """
    if margin < 0 or not all(math.isfinite(value) for value in (delta, lo, hi)):
        return "inconclusive"
    worse = lo if higher_is_worse else -hi
    better = -hi if higher_is_worse else lo
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if worse > margin:
        return "regression"
    if better > margin:
        return "improvement"
    return "inconclusive"


# ── per-window KPI extraction ────────────────────────────────────────────


def _median(values: Iterable[float]) -> float:
    usable = [value for value in values if math.isfinite(value) and value >= 0]
    return statistics.median(usable) if usable else -1.0


def _percentile(values: Iterable[float], q: float) -> float:
    usable = [value for value in values if math.isfinite(value) and value >= 0]
    return float(np.percentile(usable, q)) if usable else -1.0


def _row_group_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (
        row.get("window", ""),
        row.get("device", ""),
        row.get("phase", ""),
        int(as_float(row.get("repetition"))),
    )


def _series(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str, str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(_row_group_key(row), []).append(row)
    return groups


def _interfaces(raw_dir: Path | None) -> dict[str, dict[str, float]]:
    """Drop and packet counters per window, from the raw sidecar."""
    if raw_dir is None:
        return {}
    path = raw_dir / "stress" / "interfaces.csv"
    if not path.is_file():
        return {}
    import csv

    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            window = (row.get("window") or "").strip()
            if not window:
                continue
            out[window] = {key: as_float(value) for key, value in row.items()}
    return out


def _window_kpis(
    series: list[dict[str, str]],
    probe_rows: list[dict[str, str]],
    interface: dict[str, float] | None,
) -> dict[str, float]:
    """Every measurable KPI for one window."""
    values: dict[str, float] = {}
    values["cpu_busy_pct"] = _median(as_float(row["cpu_busy_pct"]) for row in series)
    values["cpu_iowait_pct"] = _median(as_float(row["cpu_iowait_pct"]) for row in series)
    values["mem_available_kb"] = _median(as_float(row["mem_available_kb"]) for row in series)
    values["loadavg_1m"] = _median(as_float(row["loadavg_1m"]) for row in series)

    duration = -1.0
    if series:
        duration = as_float(series[-1]["t_rel_s"]) - as_float(series[0]["t_rel_s"])
    if duration > 0 and series:
        for kpi, key in (("rx_mbps", "rx_bytes"), ("tx_mbps", "tx_bytes")):
            first = as_float(series[0][key])
            last = as_float(series[-1][key])
            if first >= 0 and last >= first:
                values[kpi] = (last - first) * 8.0 / duration / 1e6

    by_probe: dict[str, list[dict[str, str]]] = {}
    for row in probe_rows:
        by_probe.setdefault(row.get("probe", ""), []).append(row)
    lan = by_probe.get("lan")
    if lan:
        replies = [as_float(row["rtt_ms"]) for row in lan if as_float(row["rtt_ms"]) >= 0]
        values["rtt_p50_ms_lan"] = _percentile(replies, 50)
        values["rtt_p95_ms_lan"] = _percentile(replies, 95)
        values["rtt_p99_ms_lan"] = _percentile(replies, 99)
        sent = len(lan)
        if sent:
            values["rtt_loss_pct"] = (
                100.0 * sum(1 for row in lan if as_float(row["lost"]) == 1) / sent
            )
    fwd = by_probe.get("fwd")
    if fwd:
        replies = [as_float(row["rtt_ms"]) for row in fwd if as_float(row["rtt_ms"]) >= 0]
        values["rtt_p95_ms_fwd"] = _percentile(replies, 95)
    if interface:
        packets = interface.get("rx_packets", -1.0) + interface.get("tx_packets", -1.0)
        drops = interface.get("rx_drops", -1.0) + interface.get("tx_drops", -1.0)
        if packets > 0 and drops >= 0:
            values["drop_rate_pct"] = 100.0 * drops / packets
    return values


def _phase_values(
    tables_dir: Path, raw_dir: Path | None
) -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    """device -> kpi -> phase -> repetition -> value."""
    raw = read_table(tables_dir / "stress_raw.csv", "stress_raw")
    probes = read_table(tables_dir / "stress_probes.csv", "stress_probes")
    probe_groups = _series(probes)
    interfaces = _interfaces(raw_dir)
    out: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for key, rows in _series(raw).items():
        window, device, phase, repetition = key
        if not rows:
            continue
        kpis = _window_kpis(rows, probe_groups.get(key, []), interfaces.get(window))
        device_map = out.setdefault(device, {})
        for kpi, value in kpis.items():
            if not math.isfinite(value) or value < 0:
                continue
            device_map.setdefault(kpi, {}).setdefault(phase, {})[repetition] = value
    return out


def _inference_values(
    raw_dir: Path | None,
) -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    """device -> kpi -> phase -> repetition -> value, from raw/stress/inference.csv."""
    if raw_dir is None:
        return {}
    path = raw_dir / "stress" / "inference.csv"
    if not path.is_file():
        return {}
    import csv

    out: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("status") or "") != "ok":
                continue
            device = (row.get("device") or "").strip()
            phase = (row.get("phase") or "").strip()
            repetition = int(as_float(row.get("repetition")))
            for kpi in contracts.INFERENCE_KPIS:
                value = as_float(row.get(kpi))
                if not math.isfinite(value) or value < 0:
                    continue
                out.setdefault(device, {}).setdefault(kpi, {}).setdefault(phase, {})[repetition] = (
                    value
                )
    return out


# ── tables ───────────────────────────────────────────────────────────────


def _paired_between(
    by_phase: dict[str, dict[int, float]], low: str, high: str
) -> tuple[list[float], list[float]]:
    """Aligned values for two phases, paired by repetition."""
    first = by_phase.get(low, {})
    second = by_phase.get(high, {})
    common = sorted(set(first) & set(second))
    return [first[rep] for rep in common], [second[rep] for rep in common]


def _paired(by_phase: dict[str, dict[int, float]], phase: str) -> tuple[list[float], list[float]]:
    return _paired_between(by_phase, "baseline", phase)


def _impact_rows(
    device: str, kpis: dict[str, dict[str, dict[int, float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kpi in (*contracts.ROUTER_KPIS,):
        by_phase = kpis.get(kpi, {})
        if not by_phase:
            continue
        medians = {
            phase: _median(by_phase.get(phase, {}).values()) for phase in contracts.STRESS_PHASES
        }
        margin = contracts.EQUIVALENCE_MARGINS.get(kpi, -1.0)
        delta_inference, lo_inference, hi_inference = block_bootstrap(
            *_paired(by_phase, "inference")
        )
        delta_stress, lo_stress, hi_stress = block_bootstrap(*_paired(by_phase, "stress"))
        base, phase_values = _paired(by_phase, "inference")
        p_value = wilcoxon_p_value(base, phase_values)
        effect = paired_effect_size(base, phase_values)
        if kpi in HIGHER_IS_WORSE:
            higher_is_worse = True
        elif kpi in LOWER_IS_WORSE:
            higher_is_worse = False
        else:
            higher_is_worse = True
        rows.append(
            {
                "device": device,
                "kpi": kpi,
                "unit": UNITS.get(kpi, ""),
                "baseline_median": medians["baseline"],
                "inference_median": medians["inference"],
                "stress_median": medians["stress"],
                "both_median": medians["both"],
                "delta_inference": delta_inference,
                "delta_inference_lo": lo_inference,
                "delta_inference_hi": hi_inference,
                "delta_stress": delta_stress,
                "delta_stress_lo": lo_stress,
                "delta_stress_hi": hi_stress,
                "p_value": p_value,
                "effect_size": effect,
                "margin": margin,
                "verdict": verdict(
                    delta_stress, lo_stress, hi_stress, margin, higher_is_worse=higher_is_worse
                ),
            }
        )
    return rows


def _inference_rows(
    device: str, kpis: dict[str, dict[str, dict[int, float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kpi in contracts.INFERENCE_KPIS:
        by_phase = kpis.get(kpi, {})
        unloaded = _median(by_phase.get("inference", {}).values())
        loaded = _median(by_phase.get("both", {}).values())
        if unloaded < 0 and loaded < 0:
            continue
        # Unloaded is the inference-only phase; loaded is the both phase.
        delta, lo, hi = block_bootstrap(*_paired_between(by_phase, "inference", "both"))
        relative_margin = INFERENCE_REL_MARGINS.get(kpi, -1.0)
        margin = relative_margin * unloaded if unloaded > 0 and relative_margin > 0 else -1.0
        rows.append(
            {
                "device": device,
                "kpi": kpi,
                "unit": UNITS.get(kpi, "ms"),
                "unloaded_median": unloaded,
                "loaded_median": loaded,
                "delta": delta,
                "delta_lo": lo,
                "delta_hi": hi,
                "margin": margin,
                "verdict": verdict(delta, lo, hi, margin, higher_is_worse=True),
            }
        )
    return rows


def run(
    tables_dir: Path | str,
    raw_dir: Path | str | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Write ``impact.csv`` and ``inference_impact.csv``; returns the table dir."""
    say = log or _say
    table_dir = Path(tables_dir)
    raw = Path(raw_dir) if raw_dir is not None else None
    if not (table_dir / "stress_raw.csv").is_file():
        _say("analysis: no stress_raw.csv; writing header-only impact tables")
        write_table(table_dir / "impact.csv", "impact", [])
        write_table(table_dir / "inference_impact.csv", "inference_impact", [])
        return table_dir

    phase_values = _phase_values(table_dir, raw)
    inference_values = _inference_values(raw)
    impact: list[dict[str, object]] = []
    inference: list[dict[str, object]] = []
    for device in sorted(phase_values):
        impact.extend(_impact_rows(device, phase_values[device]))
    for device in sorted(inference_values):
        inference.extend(_inference_rows(device, inference_values[device]))
    if not inference:
        _say("analysis: no loadgen windows in raw/stress/inference.csv; inference table empty")

    write_table(table_dir / "impact.csv", "impact", impact)
    write_table(table_dir / "inference_impact.csv", "inference_impact", inference)
    say(
        f"analysis: impact={len(impact)} rows, inference_impact={len(inference)} rows, "
        f"draws={BOOTSTRAP_DRAWS}, seed={BOOTSTRAP_SEED}"
    )
    return table_dir
