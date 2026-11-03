"""Campaign profiles and the on-disk layout of one edge run.

A profile is the single description of *how much* a campaign measures:
models, precisions, repetitions, the core-scaling matrix, replay sample
counts, stress windows. Every other module reads the profile instead of
hard-coding counts, so "quick" and "full" differ in one place.

``smoke`` exists to prove the plumbing end-to-end on ``LocalTransport`` with a
handful of cells; ``quick`` is the interactive loop against a live router;
``full`` is the recorded campaign the report is built from.

Nothing here performs I/O beyond the small atomic table writer, which every
producer uses so a crashed cell never leaves a half-written CSV behind.
"""

from __future__ import annotations

import calendar
import csv
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from inmotion.edge import contracts

#: Phase names in execution order. ``manifest`` also runs after a failure and
#: after ``report``, so a run is identifiable even when it aborted early.
PHASES: tuple[str, ...] = (
    "preflight",
    "artifacts",
    "accuracy",
    "windows",
    "deploy",
    "replay",
    "bench",
    "power",
    "stress",
    "analyze",
    "figures",
    "report",
    "manifest",
)

#: Devices a campaign may address. ``local`` is the same code path as a router
#: with no SSH in front of it, which is what makes the smoke profile a real
#: end-to-end test rather than a mock.
LOCAL = "local"


@dataclass(frozen=True)
class Profile:
    """How much of the matrix one campaign run measures."""

    name: str
    models: tuple[str, ...]
    precisions: tuple[str, ...]
    #: Repetitions of the warm benchmark per (model, precision).
    reps: int
    #: Adaptive iterations: calibrate on ``calib_iterations``, then target
    #: ``target_seconds`` of timed loop, bounded by ``min_iterations`` /
    #: ``max_iterations`` (the contract's [30, 500] for real profiles).
    calib_iterations: int = 10
    target_seconds: float = 2.0
    min_iterations: int = 30
    max_iterations: int = 500
    warmup: int = 10
    #: Core-scaling matrix (figure F6) and concurrency matrix.
    core_models: tuple[str, ...] = ()
    core_precisions: tuple[str, ...] = ()
    core_sets: tuple[tuple[int, ...], ...] = ()
    concurrencies: tuple[int, ...] = ()
    concurrency_seconds: float = 5.0
    #: Replay sample count per device key. The full study uses 703 on device 1
    #: and a fixed stratified 200 on device 2; smoke keeps 16 so it stays tiny.
    replay_samples: Mapping[str, int] = field(default_factory=dict)
    #: Stress windows: repetitions x seconds, per device. ``0`` skips the phase.
    stress_reps: int = 0
    stress_seconds: float = 0.0
    stress_mode: str = "spin"
    stress_workers: int = 3
    #: Model/precision whose artifact the stress loadgen runs.
    stress_model: str = "lejepa"
    stress_precision: str = "fp32"
    #: Duty cycle of the stress loadgen, in milliseconds between inferences.
    #: 0 means continuous (the worst case, kept for smoke only); 2000 ms is the
    #: deployment-like rate the interference study reports.
    inference_interval_ms: int = 0

    @property
    def cells(self) -> int:
        """(model, precision, rep) benchmark cells in this profile."""
        return len(self.models) * len(self.precisions) * self.reps

    @property
    def core_cells(self) -> int:
        """Core-scaling + concurrency cells in this profile.

        The concurrency=1 row is not emitted: the 4-core serial cell already is
        that baseline, and with a real p95 rather than a loaded mean.
        """
        per_set = len(self.core_models) * len(self.core_precisions)
        concurrency_rows = sum(1 for value in self.concurrencies if value > 1)
        return per_set * (len(self.core_sets) + concurrency_rows)

    def replay_count(self, device: str) -> int:
        """Samples replayed on ``device``; 0 when the profile skips replay."""
        return int(self.replay_samples.get(device, self.replay_samples.get("default", 0)))

    def expected_rows(self) -> dict[str, int]:
        """Rows the tables should hold, for the run summary and the tests."""
        devices = len(self.replay_samples)
        load = self.cells * 2  # cold + warm per cell
        stress_windows = self.stress_reps * len(contracts.STRESS_PHASES)
        del devices
        return {
            "load": load,
            "bench": self.cells,
            "cores": self.core_cells,
            "replay": sum(self.replay_count(key) > 0 for key in self.replay_samples),
            "power": self.cells,
            "stress_windows": stress_windows * len(self.replay_samples),
        }


#: 3 precisions x 1 model is enough to exercise cold/warm load, the iteration
#: calibration, replay, power and one ABBA stress cycle, and still finishes in
#: under a minute on a laptop.
_SMOKE = Profile(
    name="smoke",
    models=("lejepa",),
    precisions=("fp32", "int8", "q4"),
    reps=1,
    target_seconds=0.25,
    min_iterations=5,
    max_iterations=50,
    warmup=2,
    replay_samples={"local": 16, "wave-router": 16, "wave-router-2": 16},
    stress_reps=1,
    stress_seconds=4.0,
    stress_workers=1,
)

#: The interactive profile: one model at three precisions, two reps, a short
#: core/concurrency matrix and the full 703/200 replay. This is what is run by
#: hand between code changes.
_QUICK = Profile(
    name="quick",
    models=("lejepa",),
    precisions=("fp32", "int8", "q4"),
    reps=2,
    core_models=("lejepa",),
    core_precisions=("fp32", "int8"),
    # `--cpus` grammar: 0, 0-1, 0-2, 0-3 are the one-, two-, three- and
    # four-core prefixes (not the pairs {0,2} and {0,3}).
    core_sets=((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)),
    concurrencies=(1, 2, 4),
    concurrency_seconds=3.0,
    replay_samples={"wave-router": 703, "wave-router-2": 200},
    stress_reps=2,
    stress_seconds=15.0,
    inference_interval_ms=2000,
)

#: The recorded campaign: the canonical five models x nine deployable
#: precisions, three repetitions, the full core set and concurrency matrix,
#: every replay sample and five one-minute stress windows per phase.
_FULL = Profile(
    name="full",
    models=contracts.MODELS,
    precisions=contracts.PRECISIONS_DEPLOYABLE,
    reps=3,
    target_seconds=2.0,
    min_iterations=30,
    max_iterations=500,
    warmup=10,
    core_models=("lejepa", "ts_jepa", "sigreg"),
    core_precisions=("fp32", "int8", "q4"),
    # `--cpus` grammar: 0, 0-1, 0-2, 0-3 are the one-, two-, three- and
    # four-core prefixes (not the pairs {0,2} and {0,3}).
    core_sets=((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)),
    concurrencies=(1, 2, 4),
    concurrency_seconds=10.0,
    replay_samples={"wave-router": 703, "wave-router-2": 200},
    stress_reps=5,
    stress_seconds=60.0,
    inference_interval_ms=2000,
)

PROFILES: dict[str, Profile] = {p.name: p for p in (_SMOKE, _QUICK, _FULL)}


def profile(name: str) -> Profile:
    """Look a profile up by name, with the valid names in the error."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown profile {name!r}; choose one of {', '.join(sorted(PROFILES))}"
        ) from None


def utc_stamp(now: float | None = None) -> str:
    """Run-directory stamp, ``YYYYmmddTHHMMSSZ``, from the UTC clock."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now if now is not None else time.time()))


@dataclass(frozen=True)
class CampaignConfig:
    """One campaign run: the profile, the devices and where everything lands."""

    profile: Profile
    devices: tuple[str, ...]
    run_dir: Path
    artifacts_dir: Path
    windows_dir: Path
    binary: Path
    force: bool = False
    resume: bool = False
    meter_file: Path | None = None
    meter_cmd: str | None = None
    source_date_epoch: int = 0
    phases: tuple[str, ...] = PHASES

    @classmethod
    def build(
        cls,
        profile_name: str,
        devices: Iterable[str],
        *,
        results_root: Path | str = Path("results/edge"),
        run_dir: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
        windows_dir: Path | str | None = None,
        binary: Path | str = Path("rust/target/x86_64-unknown-linux-musl/release/inmotion-infer"),
        force: bool = False,
        resume: bool = False,
        meter_file: Path | str | None = None,
        meter_cmd: str | None = None,
        phases: Iterable[str] | None = None,
    ) -> CampaignConfig:
        """Resolve defaults: shared artifacts live beside the runs, per-run tables inside."""
        chosen = profile(profile_name)
        keys = tuple(dict.fromkeys(devices))
        if not keys:
            raise ValueError("at least one device is required")
        for key in keys:
            if key != LOCAL and key not in contracts.DEVICES:
                known = ", ".join(("local", *contracts.DEVICES))
                raise KeyError(f"unknown device {key!r}; known: {known}")
        root = Path(results_root)
        stamp = utc_stamp()
        target = Path(run_dir) if run_dir is not None else root / "runs" / stamp
        selected = tuple(phases) if phases is not None else PHASES
        unknown = [p for p in selected if p not in PHASES]
        if unknown:
            raise KeyError(f"unknown phase(s) {', '.join(unknown)}; known: {', '.join(PHASES)}")
        return cls(
            profile=chosen,
            devices=keys,
            run_dir=target,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else root / "artifacts",
            windows_dir=Path(windows_dir) if windows_dir else root / "windows",
            binary=Path(binary),
            force=force,
            resume=resume,
            meter_file=Path(meter_file) if meter_file else None,
            meter_cmd=meter_cmd,
            source_date_epoch=int(os.environ.get("SOURCE_DATE_EPOCH") or stamp_epoch(stamp)),
            phases=selected,
        )

    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "raw"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.run_dir,
            self.tables_dir,
            self.raw_dir,
            self.figures_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def stamp_epoch(stamp: str) -> int:
    """The UTC epoch a run stamp was built from; used when SOURCE_DATE_EPOCH is unset."""
    try:
        return calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))
    except ValueError:
        return 0


def cell(value: object) -> str:
    """One CSV cell, with the repo's convention that unavailable numbers are ``-1``.

    Callers pass ``-1`` explicitly for device values that cannot be measured;
    ``None`` means "not applicable" (an error string that is empty, a skipped
    phase) and is written as an empty cell rather than a misleading zero.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-1"
        return repr(value)
    return str(value)


def write_table(path: Path | str, name: str, rows: Iterable[Mapping[str, object]]) -> Path:
    """Write one canonical table atomically.

    The header comes from ``contracts.COLUMNS`` and unknown keys are dropped, so
    a producer cannot drift the schema by accident. The temporary file lives in
    the destination directory, so the final ``os.replace`` is atomic on POSIX.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = contracts.COLUMNS[name]
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: cell(row.get(key)) for key in header})
    os.replace(tmp, target)
    return target


def read_table(path: Path | str, name: str) -> list[dict[str, str]]:
    """Read one canonical table back, asserting the header it found.

    ``-1`` is kept as-is rather than coerced: the readers decide whether a KPI
    is available, and silently turning it into 0.0 would fabricate a result.
    """
    target = Path(path)
    header = contracts.COLUMNS[name]
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        found = tuple(reader.fieldnames or ())
        if found != header:
            raise ValueError(f"{target}: expected columns {header}, found {found}")
        return list(reader)


def as_float(value: object, default: float = -1.0) -> float:
    """Parse a table cell, mapping empty and malformed values to ``default``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
