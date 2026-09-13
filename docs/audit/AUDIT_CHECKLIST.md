# Audit checklist

What a reviewer can independently verify, and how. Every row names a command or
a file. Nothing here relies on trusting the authors.

## 1. Can the claimed models be loaded?

| check | command | expected |
|---|---|---|
| every registered model strict-loads | `uv run inmotion model verify` | `All 18 model(s) verified.` |
| parameter counts match the registry | same command | per-model `OK` |
| a model's output has not silently changed | `uv run pytest tests/test_model_outputs_golden.py` | pass |

`load_state_dict(..., strict=False)` is banned. It had been hiding a model that
was missing 13 keys and therefore predicting partly from random weights.

## 2. Are the datasets what they claim to be?

| check | command | expected |
|---|---|---|
| checksums and row counts | `uv run inmotion datasets verify` | `All 7 dataset version(s) match` |
| lineage and label provenance | `uv run inmotion datasets show <id>` | lineage, audit status, first commit |
| untrustworthy versions warn | `uv run inmotion datasets resolve ds-pure` | warning on stderr |

## 3. Is the reported number reproducible?

`docs/isac/results/MASTER_RESULTS.md` maps each reported value to the artifact
that produced it. `docs/reproducibility.md` states the split, seed and metric and
lists what is *not* reproducible, with reasons.

The three `mamba3_*` parameter counts and the Mamba-3 HPO search space are known
and documented discrepancies, not silent ones. A test fails if either is closed
by editing the wrong side.

## 4. Is the code quality real?

| check | command | expected |
|---|---|---|
| lint | `uv run ruff check src dl ml_classification tests` | `All checks passed!` |
| formatting | `uv run ruff format --check src dl ml_classification tests` | no drift |
| tests | `uv run pytest tests/` | pass |
| demo tests | `cd demo && uv run --project . pytest` | pass |
| reference docs current | `uv run python scripts/docs/generate_reference.py --check` | up to date |
| no committed secrets | `uv run pytest tests/test_no_secrets.py` | pass |

All of the above run together via `./scripts/check.sh`. There is no CI by
choice, so run it before pushing.

## 5. Are hidden failure modes covered by tests?

| risk | test |
|---|---|
| an architecture drifts but still loads | `tests/test_checkpoint_contract.py` (strict load + exact parameter count) |
| a forward pass changes silently | `tests/test_model_outputs_golden.py` (frozen fingerprint) |
| an adopted checkpoint is wrong | `tests/test_adoption.py` (strict-load assertion per recovered spec) |
| a config changes between processes | `tests/test_config_reproducibility.py` |
| a credential is committed | `tests/test_no_secrets.py` |
| the CLI breaks | `tests/test_cli.py` |
| reference docs drift from the code | `scripts/docs/generate_reference.py --check` |

## 6. Is the structure navigable?

| starting point | contents |
|---|---|
| `README.md` | what the project is, how to run it |
| `docs/README.md` | documentation index |
| `docs/architecture.md` | how the pieces fit together |
| `docs/reference/modules.md` | every module and its public API, generated from source |
| `docs/reference/cli.md` | every command and option, generated from the app |
| `docs/reference/models.md` | every model, generated from the registry |
| `docs/reference/datasets.md` | every dataset version, generated from the manifest |

The reference documents are generated, so they cannot disagree with the code.
`--check` fails when they are stale.

## 7. Data and code availability

| item | status |
|---|---|
| code | this repository |
| dataset | IEEE Dataport DOI `10.21227/55nm-0r91`; a copy is committed under `data/processed/` |
| model weights | gitignored, present in `backup/` and `20-aug/models/`; `inmotion model inventory` recovers their architectures |
| licence | not yet declared — **add one before submission** |
| citation | `CITATION.cff` present; replace the placeholder reference with the camera-ready citation on acceptance |

## Open items

These are known and not hidden:

1. No `LICENSE` yet — add one before submission (the status table above records
   the same; no `LICENSE`/`LICENCE`/`COPYING` file exists at the repository root).
   `CITATION.cff` is present and tracked (added in `16f45fd`); only its
   `references` block is still the placeholder, to be replaced with the
   camera-ready citation on acceptance.
3. `ds-icaisf` label provenance needs confirmation; if it belongs to the older
   lineage the reported ensemble numbers are affected.
5. The classical `src/inmotion/classification/` package has not moved under
   `src/inmotion/`, so two top-level packages coexist.
6. `run_dl.py` and `run_exotic.py` remain large single modules (now
   `pipelines/dl.py` and `pipelines/exotic.py`); splitting them further is
   mechanical but not free of regression risk.
