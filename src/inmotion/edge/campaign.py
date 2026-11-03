"""Campaign orchestration: one run directory, resumable phases, a manifest.

A campaign is a folder that can be handed to someone else and re-derived:

    results/edge/runs/<UTC>/
        manifest.json   provenance: git, lockfiles, binary, devices, versions
        config.json     the profile and the paths the run used
        state.json      per-phase status, so an interrupted run can resume
        report.md       the run's own summary
        raw/            device roots and per-window raw output
        tables/         the canonical CSVs the figures read
        figures/        the eight figures
        logs/           the campaign log

Phases run in the contract's order and each one is idempotent: a phase that
finished is skipped on ``--resume``, and one that did not is redone in full.
A finished run is never silently overwritten — that needs ``--force``.

Device tables are staged per device (``raw/stage/<device>/tables``) and merged
into the run's ``tables/`` after each phase, because two routers write the same
table names with a ``device`` column that keeps them apart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

from inmotion.edge import contracts, transport
from inmotion.edge.config import (
    LOCAL,
    CampaignConfig,
    Profile,
    read_table,
    write_table,
)

#: Tables the accuracy workstream owns and that figures also need in the run.
_ACCURACY_TABLES: tuple[str, ...] = ("accuracy", "per_class", "confusion")

_DEFAULT_BINARY = Path("rust/target/x86_64-unknown-linux-musl/release/inmotion-infer")
_DEFAULT_ROOT = Path("results/edge")


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = transport.redact(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=15, cwd=Path.cwd()
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _versions() -> dict[str, str]:
    import inmotion

    versions = {
        "inmotion": getattr(inmotion, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for tool in ("uv", "cargo", "rustc"):
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=10)
            versions[tool] = result.stdout.strip() or result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired):
            versions[tool] = "unavailable"
    return versions


def _lockfiles() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("uv.lock", "rust/Cargo.lock", "pyproject.toml"):
        path = Path(name)
        if path.is_file():
            hashes[name] = _sha256(path)
    return hashes


# ── state ────────────────────────────────────────────────────────────────


def _load_state(config: CampaignConfig) -> dict[str, object]:
    if not config.state_path.is_file():
        return {"run": str(config.run_dir), "started": _now(), "phases": {}}
    try:
        return json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"run": str(config.run_dir), "started": _now(), "phases": {}}


def _phase_status(state: dict[str, object], name: str) -> str:
    phases = state.get("phases", {})
    entry = phases.get(name, {}) if isinstance(phases, dict) else {}
    return str(entry.get("status", "")) if isinstance(entry, dict) else ""


def _record(
    config: CampaignConfig,
    state: dict[str, object],
    name: str,
    status: str,
    detail: str,
    started: str,
) -> None:
    phases = state.setdefault("phases", {})
    assert isinstance(phases, dict)
    phases[name] = {
        "status": status,
        "detail": detail,
        "started": started,
        "finished": _now(),
    }
    state["updated"] = _now()
    _json_write(config.state_path, state)


# ── runtime helpers ──────────────────────────────────────────────────────


@dataclass
class _Runtime:
    config: CampaignConfig
    log: Callable[[str], None]
    state: dict[str, object]
    transports: dict[str, transport.DeviceTransport] = field(default_factory=dict)

    def device(self, key: str) -> transport.DeviceTransport:
        if key not in self.transports:
            if key == LOCAL:
                root = self.config.raw_dir / "local" / "device"
                self.transports[key] = transport.open_device(key, root=root)
            else:
                self.transports[key] = transport.open_device(key)
        return self.transports[key]

    def stage_tables(self, key: str) -> Path:
        path = self.config.raw_dir / "stage" / key / "tables"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def device_raw(self, key: str) -> Path:
        path = self.config.raw_dir / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        for device in self.transports.values():
            close = getattr(device, "close", None)
            if callable(close):
                close()
        self.transports.clear()

    def merge_table(self, name: str) -> None:
        rows: list[dict[str, str]] = []
        for key in self.config.devices:
            path = self.stage_tables(key) / f"{name}.csv"
            if path.is_file():
                rows.extend(read_table(path, name))
        target = write_table(self.config.tables_dir / f"{name}.csv", name, rows)
        self.log(f"campaign: merged {name} ({len(rows)} rows) -> {target}")


def _shared_root(config: CampaignConfig) -> Path:
    """The campaign-wide directory that holds artifacts, windows and shared tables."""
    return config.artifacts_dir.parent


def _artifact_matrix(profile: Profile) -> list[tuple[str, str]]:
    return [(model, precision) for model in profile.models for precision in profile.precisions]


# ── phases ───────────────────────────────────────────────────────────────


def _phase_preflight(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    if not config.binary.is_file():
        raise FileNotFoundError(
            f"runtime binary not found at {config.binary}; build it with "
            "`cargo build --release --target x86_64-unknown-linux-musl -p inmotion-cli`"
        )
    state = runtime.state
    devices = state.setdefault("devices", {})
    assert isinstance(devices, dict)
    for key in config.devices:
        device = runtime.device(key)
        facts = device.preflight()
        facts["binary_sha256"] = transport.binary_sha256(config.binary)
        devices[key] = facts
        runtime.log(
            f"campaign: preflight {key}: kernel={facts.get('kernel')} "
            f"cores={facts.get('cores')} root_ok={facts.get('root_ok')}"
        )
    _json_write(config.state_path, state)
    return "done", f"{len(config.devices)} device(s) reachable"


def _phase_artifacts(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import bench

    config = runtime.config
    profile = config.profile
    wanted = _artifact_matrix(profile)
    present = bench.resolve_artifacts(config.artifacts_dir)
    missing = [pair for pair in wanted if pair not in present]
    if not missing:
        return "skipped", f"{len(wanted)} artifact(s) already present"
    from inmotion.edge import accuracy

    accuracy.export_artifacts(
        config.artifacts_dir,
        models=tuple(dict.fromkeys(model for model, _p in wanted)),
        precisions=tuple(dict.fromkeys(precision for _m, precision in wanted)),
    )
    after = bench.resolve_artifacts(config.artifacts_dir)
    still = [pair for pair in wanted if pair not in after]
    if still:
        raise RuntimeError(
            "artifacts missing after export: "
            + ", ".join(f"{model}-{precision}" for model, precision in still[:8])
        )
    return "done", f"exported {len(missing)} artifact(s)"


def _phase_accuracy(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    target = config.tables_dir
    target.mkdir(parents=True, exist_ok=True)
    shared = _shared_root(config) / "tables"
    reused = 0
    for name in _ACCURACY_TABLES:
        destination = target / f"{name}.csv"
        source = shared / f"{name}.csv"
        if destination.is_file():
            continue
        if source.is_file():
            shutil.copy2(source, destination)
            reused += 1
    if all((target / f"{name}.csv").is_file() for name in _ACCURACY_TABLES):
        return "skipped", f"reused {reused} shared accuracy table(s)"
    from inmotion.edge import accuracy

    accuracy.evaluate_artifacts(config.artifacts_dir, target)
    return "done", "scored the artifact matrix"


def _phase_windows(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    models = tuple(dict.fromkeys(config.profile.models))
    missing = [
        model for model in models if not (config.windows_dir / model / "windows.txt").is_file()
    ]
    if not missing:
        labels = sum(
            len((config.windows_dir / model / "labels.txt").read_text().splitlines())
            for model in models
            if (config.windows_dir / model / "labels.txt").is_file()
        )
        return "skipped", f"windows present for {len(models)} model(s), {labels} label rows"
    from inmotion.edge import accuracy

    accuracy.export_windows(config.windows_dir, models=models)
    return "done", f"exported windows for {', '.join(missing)}"


def _phase_deploy(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import bench

    config = runtime.config
    detail: list[str] = []
    for key in config.devices:
        device = runtime.device(key)
        remote = bench.deploy_binary(device, config.binary, log=runtime.log)
        detail.append(f"{key}:{remote}")
    return "done", ", ".join(detail)


def _phase_replay(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import replay

    config = runtime.config
    for key in config.devices:
        replay.run(
            runtime.device(key),
            config.artifacts_dir,
            config.windows_dir,
            runtime.stage_tables(key),
            runtime.device_raw(key),
            config.profile,
            binary=config.binary,
            accuracy_csv=config.tables_dir / "accuracy.csv",
            log=runtime.log,
        )
    runtime.merge_table("replay")
    rows = read_table(config.tables_dir / "replay.csv", "replay")
    passed = sum(1 for row in rows if row.get("parity_ok") == "True")
    return "done", f"{len(rows)} cell(s), {passed} parity-ok"


def _phase_bench(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import bench

    config = runtime.config
    for key in config.devices:
        bench.run(
            runtime.device(key),
            config.artifacts_dir,
            runtime.stage_tables(key),
            runtime.device_raw(key),
            config.profile,
            binary=config.binary,
            log=runtime.log,
        )
    for name in ("load", "bench", "cores"):
        runtime.merge_table(name)
    return "done", f"{config.profile.cells} cell(s) x {len(config.devices)} device(s)"


def _phase_power(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import power

    config = runtime.config
    for key in config.devices:
        power.run(
            runtime.device(key),
            runtime.stage_tables(key),
            meter_file=config.meter_file,
            meter_cmd=config.meter_cmd,
            log=runtime.log,
        )
    runtime.merge_table("power")
    rows = read_table(config.tables_dir / "power.csv", "power")
    sources = sorted({row.get("source", "") for row in rows})
    return "done", f"{len(rows)} row(s), source(s) {', '.join(sources) or 'none'}"


def _phase_stress(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import stress

    config = runtime.config
    if config.profile.stress_reps <= 0:
        for name in ("stress_raw", "stress_probes"):
            write_table(config.tables_dir / f"{name}.csv", name, [])
        return "skipped", "profile does not run the interference study"
    for key in config.devices:
        stress.run(
            runtime.device(key),
            config.artifacts_dir,
            runtime.stage_tables(key),
            runtime.device_raw(key),
            config.profile,
            binary=config.binary,
            token=f"{config.run_dir.name}-{key}",
            log=runtime.log,
        )
    for name in ("stress_raw", "stress_probes"):
        runtime.merge_table(name)
    _merge_sidecars(runtime)
    raw_rows = read_table(config.tables_dir / "stress_raw.csv", "stress_raw")
    probe_rows = read_table(config.tables_dir / "stress_probes.csv", "stress_probes")
    return "done", f"{len(raw_rows)} sample row(s), {len(probe_rows)} probe row(s)"


def _merge_sidecars(runtime: _Runtime) -> None:
    """Combine the per-device raw stress sidecars analysis reads."""
    config = runtime.config
    target_dir = config.raw_dir / "stress"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, header in (
        (
            "inference.csv",
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
        ),
        (
            "interfaces.csv",
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
        ),
    ):
        rows: list[dict[str, object]] = []
        for key in config.devices:
            source = config.raw_dir / key / "stress" / name
            if not source.is_file():
                continue
            with source.open(newline="", encoding="utf-8") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        _write_raw_csv(target_dir / name, header, rows)


def _write_raw_csv(path: Path, header: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, -1) for key in header})
    os.replace(tmp, path)


def _phase_analyze(runtime: _Runtime) -> tuple[str, str]:
    from inmotion.edge import analysis

    config = runtime.config
    analysis.run(config.tables_dir, config.raw_dir, log=runtime.log)
    impact = read_table(config.tables_dir / "impact.csv", "impact")
    inference = read_table(config.tables_dir / "inference_impact.csv", "inference_impact")
    return "done", f"impact={len(impact)} row(s), inference_impact={len(inference)} row(s)"


def _phase_figures(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    if not any(key != LOCAL for key in config.devices):
        # F4-F8 plot router rows; a local-only smoke run has none and the figure
        # builder refuses a half-empty panel by design. Skipping is honest; the
        # accuracy tables were still produced and are what smoke proves.
        return "skipped", "local-only run: figures need router device rows"
    if find_spec("inmotion.edge.figures") is None:
        return "skipped", "figures module not available"
    from inmotion.edge import figures

    paths = figures.build(config.tables_dir, config.figures_dir)
    return "done", f"{len(paths)} file(s) in {config.figures_dir}"


def _phase_report(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    out = build_report(config.tables_dir, config.run_dir / "report.md", config=config)
    return "done", str(out)


def _phase_manifest(runtime: _Runtime) -> tuple[str, str]:
    config = runtime.config
    payload = manifest(runtime)
    _json_write(config.manifest_path, payload)
    return "done", str(config.manifest_path)


PHASE_HANDLERS: dict[str, Callable[[_Runtime], tuple[str, str]]] = {
    "preflight": _phase_preflight,
    "artifacts": _phase_artifacts,
    "accuracy": _phase_accuracy,
    "windows": _phase_windows,
    "deploy": _phase_deploy,
    "replay": _phase_replay,
    "bench": _phase_bench,
    "power": _phase_power,
    "stress": _phase_stress,
    "analyze": _phase_analyze,
    "figures": _phase_figures,
    "report": _phase_report,
    "manifest": _phase_manifest,
}


# ── report and manifest ──────────────────────────────────────────────────


def _table_rows(tables_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in contracts.COLUMNS:
        path = tables_dir / f"{name}.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            counts[name] = max(0, sum(1 for _ in handle) - 1)
    return counts


def build_report(
    tables_dir: Path | str,
    out_path: Path | str,
    *,
    config: CampaignConfig | None = None,
    extra: Sequence[str] = (),
) -> Path:
    """Write a compact, deterministic summary of the run's tables."""
    table_dir = Path(tables_dir)
    target = Path(out_path)
    lines: list[str] = ["# Edge campaign run", ""]
    if config is not None:
        lines += [
            f"- profile: `{config.profile.name}`",
            f"- devices: {', '.join(f'`{key}`' for key in config.devices)}",
            f"- run dir: `{config.run_dir}`",
            f"- source date epoch: `{config.source_date_epoch}`",
            "",
        ]
    counts = _table_rows(table_dir)
    lines += ["## Tables", "", "| table | rows |", "| --- | ---: |"]
    for name, count in sorted(counts.items()):
        lines.append(f"| `{name}.csv` | {count} |")
    lines.append("")

    accuracy_path = table_dir / "accuracy.csv"
    if accuracy_path.is_file():
        rows = read_table(accuracy_path, "accuracy")
        fp32 = [row for row in rows if row.get("precision") == "fp32" and row.get("mcc")]
        if fp32:
            lines += ["## Accuracy (fp32)", "", "| model | MCC |", "| --- | ---: |"]
            for row in sorted(fp32, key=lambda item: item["model"]):
                lines.append(f"| {row['model']} | {float(row['mcc']):.4f} |")
            lines.append("")

    replay_path = table_dir / "replay.csv"
    if replay_path.is_file():
        rows = read_table(replay_path, "replay")
        usable = [row for row in rows if row.get("status") == "ok"]
        if usable:
            best = min(usable, key=lambda row: float(row["agreement"]) if row["agreement"] else 1.0)
            lines += [
                "## On-device replay",
                "",
                f"- cells: {len(rows)} ({len(usable)} ok)",
                f"- parity ok: {sum(1 for row in rows if row.get('parity_ok') == 'True')}",
                f"- lowest agreement: {best['agreement']} ({best['model']}-{best['precision']}, "
                f"{best['device']})",
                "",
            ]

    bench_path = table_dir / "bench.csv"
    if bench_path.is_file():
        rows = [row for row in read_table(bench_path, "bench") if row.get("status") == "ok"]
        if rows:
            by_device: dict[str, list[float]] = {}
            for row in rows:
                if float(row["p95_ms"]) > 0:
                    by_device.setdefault(row["device"], []).append(float(row["p95_ms"]))
            lines += ["## Latency (p95, ms)", ""]
            for device, values in sorted(by_device.items()):
                lines.append(f"- {device}: best {min(values):.3f}, worst {max(values):.3f}")
            lines.append("")

    power_path = table_dir / "power.csv"
    if power_path.is_file():
        rows = read_table(power_path, "power")
        if rows:
            sources = sorted({row.get("source", "") for row in rows})
            lines += ["## Energy", "", f"- sources: {', '.join(sources)}", ""]

    impact_path = table_dir / "impact.csv"
    if impact_path.is_file():
        rows = read_table(impact_path, "impact")
        if rows:
            counts_by_verdict: dict[str, int] = {}
            for row in rows:
                counts_by_verdict[row.get("verdict", "")] = (
                    counts_by_verdict.get(row.get("verdict", ""), 0) + 1
                )
            lines += [
                "## Router impact",
                "",
                ", ".join(
                    f"{count} {verdict}" for verdict, count in sorted(counts_by_verdict.items())
                ),
                "",
            ]

    if extra:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in extra]
        lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(transport.redact("\n".join(lines).rstrip() + "\n"), encoding="utf-8")
    os.replace(tmp, target)
    return target


def manifest(runtime: _Runtime) -> dict[str, object]:
    """Provenance for the run: no secret can survive ``transport.redact``."""
    config = runtime.config
    state = runtime.state
    devices = state.get("devices", {}) if isinstance(state.get("devices"), dict) else {}
    artifacts_manifest = config.artifacts_dir / "artifacts.csv"
    figures = config.figures_dir.rglob("*") if config.figures_dir.is_dir() else ()
    return {
        "run": {
            "dir": str(config.run_dir),
            "profile": config.profile.name,
            "devices": list(config.devices),
            "started": state.get("started"),
            "finished": _now(),
            "source_date_epoch": config.source_date_epoch,
        },
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "lockfiles": _lockfiles(),
        "binary": {
            "path": str(config.binary),
            "sha256": transport.binary_sha256(config.binary) if config.binary.is_file() else "",
            "bytes": config.binary.stat().st_size if config.binary.is_file() else -1,
        },
        "artifacts": {
            "dir": str(config.artifacts_dir),
            "count": len(list(config.artifacts_dir.glob("*-*"))),
            "manifest_sha256": _sha256(artifacts_manifest) if artifacts_manifest.is_file() else "",
        },
        "devices": devices,
        "versions": _versions(),
        "tables": _table_rows(config.tables_dir),
        "figures": {
            str(path.relative_to(config.figures_dir)): _sha256(path)
            for path in sorted(figures)
            if path.is_file()
        },
        "phases": state.get("phases", {}),
    }


# ── entry point ──────────────────────────────────────────────────────────


def run(
    profile_name: str,
    devices: Iterable[str],
    *,
    run_dir: Path | str | None = None,
    resume: bool = False,
    force: bool = False,
    results_root: Path | str = _DEFAULT_ROOT,
    artifacts_dir: Path | str | None = None,
    windows_dir: Path | str | None = None,
    binary: Path | str = _DEFAULT_BINARY,
    meter_file: Path | str | None = None,
    meter_cmd: str | None = None,
    phases: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> CampaignConfig:
    """Create (or resume) a run and execute the requested phases."""
    say = log or _say
    config = CampaignConfig.build(
        profile_name,
        devices,
        results_root=results_root,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        windows_dir=windows_dir,
        binary=binary,
        force=force,
        resume=resume,
        meter_file=meter_file,
        meter_cmd=meter_cmd,
        phases=phases,
    )
    if config.run_dir.exists():
        finished = _finished(config)
        if force:
            shutil.rmtree(config.run_dir)
        elif resume or not any(config.run_dir.iterdir()):
            pass
        elif finished:
            raise FileExistsError(
                f"{config.run_dir} already holds a finished run; pass --force to overwrite "
                "or --resume to continue it"
            )
        else:
            raise FileExistsError(
                f"{config.run_dir} already exists; pass --resume to continue it or --force "
                "to start over"
            )
    config.ensure_dirs()
    state = _load_state(config)
    runtime = _Runtime(config=config, log=say, state=state)
    _json_write(
        config.run_dir / "config.json",
        {
            "profile": config.profile.name,
            "devices": list(config.devices),
            "artifacts_dir": str(config.artifacts_dir),
            "windows_dir": str(config.windows_dir),
            "binary": str(config.binary),
            "phases": list(config.phases),
            "force": config.force,
            "resume": config.resume,
            "meter_file": str(config.meter_file) if config.meter_file else None,
            "meter_cmd": config.meter_cmd,
        },
    )
    say(f"campaign: run dir {config.run_dir}")
    failed: list[str] = []
    fatal = ""
    tolerated = {"figures", "report", "manifest"}
    try:
        for name in config.phases:
            handler = PHASE_HANDLERS[name]
            status = _phase_status(state, name)
            if resume and status == "done":
                say(f"campaign: {name}: done in a previous attempt, skipping")
                continue
            started = _now()
            say(f"campaign: {name}: start")
            try:
                result, detail = handler(runtime)
            except Exception as exc:  # noqa: BLE001 - recorded per phase, then reflected in state
                result, detail = "failed", f"{type(exc).__name__}: {exc}"
                failed.append(name)
                say(f"campaign: {name}: FAILED {detail}")
                if name not in tolerated:
                    fatal = f"phase {name} failed: {detail}"
                    _record(config, state, name, result, detail, started)
                    break
            else:
                say(f"campaign: {name}: {result} ({detail})")
            _record(config, state, name, result, detail, started)
        _update_latest(config, say)
        state["status"] = "complete" if not failed else "incomplete"
        state["finished"] = _now()
        _json_write(config.state_path, state)
        _json_write(config.manifest_path, manifest(runtime))
    finally:
        runtime.close()
    if fatal:
        raise RuntimeError(f"campaign: {fatal}")
    if failed:
        raise RuntimeError(f"campaign finished with failed phase(s): {', '.join(failed)}")
    say(f"campaign: report {config.run_dir / 'report.md'}")
    say(f"campaign: manifest {config.manifest_path}")
    return config


def _finished(config: CampaignConfig) -> bool:
    try:
        state = json.loads(config.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("status") == "complete"


def _update_latest(config: CampaignConfig, say: Callable[[str], None]) -> None:
    """Point ``results/edge/latest`` at this run, atomically."""
    root = config.artifacts_dir.parent
    if not root.is_dir():
        return
    link = root / "latest"
    target = os.path.relpath(config.run_dir, root)
    temporary = root / ".latest.tmp"
    try:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
        os.symlink(target, temporary)
        os.replace(temporary, link)
        say(f"campaign: latest -> {target}")
    except OSError:
        say(f"campaign: could not update {link}")
