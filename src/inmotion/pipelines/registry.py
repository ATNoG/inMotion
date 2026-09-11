"""Pipeline registry.

Each pipeline is an existing module with a ``main()`` that reads ``sys.argv``.
:func:`run` is the single place that invokes one, so the CLI does not need to
know how any particular pipeline is wired.

Pipelines are run **in process** with ``sys.argv`` temporarily replaced, rather
than by spawning a subprocess. That keeps tracebacks readable, lets a debugger
and a profiler attach normally, and means the CLI does not depend on the current
working directory containing a script file.

Every pipeline still has its own module-level ``__main__`` guard, so
``uv run python -m inmotion.pipelines.exotic --help`` also works.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Pipeline:
    """One runnable pipeline."""

    name: str
    module: str
    summary: str
    """One line, shown by `inmotion run list`."""

    @property
    def dotted(self) -> str:
        return f"inmotion.{self.module}"


PIPELINES: tuple[Pipeline, ...] = (
    Pipeline("classification", "pipelines.classification",
             "classical ML classifiers (EDA, CV, Optuna, feature importance)"),
    Pipeline("dl", "pipelines.dl",
             "supervised deep learning: baselines, ensembles, MoE, deep stacking, NAS"),
    Pipeline("exotic", "pipelines.exotic",
             "JEPA world models and SIGReg: SSL pretraining + fine-tuning"),
    Pipeline("ensemble", "pipelines.ensemble",
             "greedy ensemble selection over existing checkpoints"),
    Pipeline("mega-ensemble", "pipelines.mega_ensemble",
             "OOF stacking over every model family, with calibration"),
    Pipeline("hpo-paper", "pipelines.hpo_paper",
             "retrain the paper's HPO-best configs (Optuna DBs were deleted)"),
    Pipeline("evaluate-checkpoints", "evaluation.checkpoints",
             "retroactive per-class metrics for saved DL checkpoints"),
    Pipeline("shap", "evaluation.shap",
             "SHAP channel attribution for the world models"),
    Pipeline("feature-importance", "evaluation.feature_importance",
             "permutation feature importance for DL models"),
    Pipeline("analyze-seeds", "analysis.seeds",
             "combine and plot multi-seed results"),
    Pipeline("analyze-interference", "analysis.interference",
             "cross-route interference analysis from concurrent_noise_path"),
    Pipeline("regenerate-plots", "viz.regenerate",
             "rebuild classical-ML figures from saved CSVs"),
    Pipeline("regenerate-dl-plots", "viz.regenerate_dl",
             "rebuild DL figures from saved CSVs"),
)

BY_NAME: dict[str, Pipeline] = {p.name: p for p in PIPELINES}


def get(name: str) -> Pipeline:
    """Look up a pipeline, or raise with the available names."""
    try:
        return BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(BY_NAME))
        raise KeyError(f"unknown pipeline {name!r}; available: {known}") from exc


@contextlib.contextmanager
def _argv(argv: list[str]) -> Iterator[None]:
    """Temporarily replace ``sys.argv``, restoring it afterwards."""
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def run(pipeline: Pipeline | str, args: list[str] | None = None) -> int:
    """Run a pipeline with ``args``, returning its exit code.

    A pipeline whose ``main()`` returns None is treated as success.
    """
    if isinstance(pipeline, str):
        pipeline = get(pipeline)

    module = importlib.import_module(pipeline.dotted)
    entry = getattr(module, "main", None)
    if entry is None:
        raise AttributeError(f"{pipeline.dotted} has no main() to call")

    with _argv([pipeline.dotted, *(args or [])]):
        result = entry()
    return 0 if result is None else int(result)
