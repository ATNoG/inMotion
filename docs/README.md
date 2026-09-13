# Documentation

Start with `architecture.md` for how the system fits together, then whichever
reference you need. `../README.md` is the project front page.

## Orientation

| document | what it answers |
|---|---|
| [`architecture.md`](architecture.md) | how data flows, what the model families are, how the registry and sidecars work |
| [`reproducibility.md`](reproducibility.md) | environment, seeds, split convention, what is and is not reproducible, known provenance gaps |

## Guides

| document | what it answers |
|---|---|
| [`guides/commands.md`](guides/commands.md) | every command, when to use it, and a realistic invocation |
| [`guides/running-models.md`](guides/running-models.md) | how to use a checkpoint: list, inspect, verify, predict, adopt |
| [`guides/rust-and-devices.md`](guides/rust-and-devices.md) | how to run inference without Python: export an artifact, build for another device, prove parity on it, and measure it there |
| [`guides/training.md`](guides/training.md) | how to train each family, on a workstation or a cluster |

## Reference (generated from the code)

Regenerate with `uv run python scripts/docs/generate_reference.py`. The docs
freshness check in `scripts/check.sh` fails if they are stale.

| document | contents |
|---|---|
| [`reference/cli.md`](reference/cli.md) | every command, argument, option and default |
| [`reference/pipelines.md`](reference/pipelines.md) | every pipeline option, read from its config dataclass |
| [`reference/models.md`](reference/models.md) | every registered model: factory, architecture, parameter count, provenance |
| [`reference/modules.md`](reference/modules.md) | every module, its purpose and its public API |
| [`reference/datasets.md`](reference/datasets.md) | every dataset version: checksum, counts, lineage, label audit |

## Audit

| document | contents |
|---|---|
| [`audit/AUDIT_CHECKLIST.md`](audit/AUDIT_CHECKLIST.md) | what a reviewer can independently verify, and the command for each |
| [`audit/MODEL_CARDS.md`](audit/MODEL_CARDS.md) | per-family model cards: purpose, data, evaluation, limitations, out-of-scope use |

## Papers and results

| document | contents |
|---|---|
| [`isac/results/MASTER_RESULTS.md`](isac/results/MASTER_RESULTS.md) | the single source of truth for reported numbers, each mapped to its artifact |
| `eucnc/`, `icaisf/`, `isac/` | the LaTeX sources, frozen |
| [`supplementary/`](supplementary/) | hyperparameter search spaces and optima |
| [`../ENSEMBLE_TECHNIQUES_REPORT.md`](ENSEMBLE_TECHNIQUES_REPORT.md) | the literature basis for the ensemble combiner |

## Working notes and history

| document | contents |
|---|---|
| [`reports/ARTIFACTS.md`](reports/ARTIFACTS.md) | where models and results live, what is tracked, and why |
| [`reports/MIGRATION.md`](reports/MIGRATION.md) | old command → new command, and what remains to reorganize |
| [`reports/REFACTOR_SUMMARY.md`](reports/REFACTOR_SUMMARY.md) | what changed in the reorganization, with before/after measurements |
| [`reports/`](reports/) | earlier analysis reports, kept for provenance: `AI_RESULT.md` (classical ML), `ROADMAP_TO_90.md`, `IMPLEMENTATION_SUMMARY.md`, `research_sources.md` |
