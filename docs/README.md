# Documentation

What is in this directory, what each document means, and which one to open for
which question. `../README.md` is the project front page.

`docs/` holds **four different kinds of material** that are easy to confuse:

| kind                                         | directories                                                      | trust it for                                     |
| -------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| **How the system works and how to drive it** | `architecture.md`, `reproducibility.md`, `guides/`, `reference/` | running the code, and what the interfaces accept |
| **What was checked, by whom, and how**       | `audit/`, `reports/`                                             | whether a claim has been independently verified  |
| **The published research**                   | `isac/`, `eucnc/`, `icaisf/`, `supplementary/`                   | the papers, their numbers and their sources      |
| **Project context**                          | `eucnc/project_description.md`                                   | why this exists, in funder terms                 |

The distinction matters when you are deciding what to believe. A guide tells you
what a command _accepts_; a verification report tells you what someone _ran_; a
master-results file tells you which artifact a published number _came from_.
They are not interchangeable, and only one of the three is generated from code.

---

## Start here

| if you are…                            | read                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| new to the codebase                    | [`architecture.md`](architecture.md) — the data flow and the model families                                     |
| trying to run something                | [`guides/commands.md`](guides/commands.md) — every command with a realistic invocation                          |
| looking for an exact option or default | [`reference/cli.md`](reference/cli.md) — generated from the program                                             |
| reviewing the work                     | [`audit/AUDIT_CHECKLIST.md`](audit/AUDIT_CHECKLIST.md) — what you can verify yourself, and the command for each |
| taking a model to another machine      | [`guides/rust-and-devices.md`](guides/rust-and-devices.md)                                                      |

---

## 1. How the system works

| document                                   | what it means                                                                                                                                                                                 |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`architecture.md`](architecture.md)       | The end-to-end story: WiFi RSSI windows → classification → the model families and how they differ; the registry, sidecars and artifact store. Read first.                                     |
| [`reproducibility.md`](reproducibility.md) | What an independent reader needs to re-run this work: environment, seeds, the split convention, **what is and is not reproducible**, and the known provenance gaps stated rather than hidden. |

## 2. Doing the work — guides

Task-oriented, hand-written. Each answers "what do I type, and why".

| document                                                   | what it means                                                                                                                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`guides/commands.md`](guides/commands.md)                 | Every command, when to reach for it, and a realistic invocation. The map of the whole CLI.                                                                                           |
| [`guides/training.md`](guides/training.md)                 | How to train each family, on a workstation or a cluster, including the environment traps (for example `MAMBA_SSM_AVAILABLE=0`).                                                      |
| [`guides/running-models.md`](guides/running-models.md)     | Using a saved checkpoint: list, inspect, verify, predict, adopt one that has no sidecar, and the known exceptions.                                                                   |
| [`guides/rust-and-devices.md`](guides/rust-and-devices.md) | Running inference **without Python**: export an artifact, cross-build for another device, prove parity on that device, and measure it. Covers the before/after kernel benchmark too. |

## 3. Looking things up — reference (generated)

These five are **generated from the code** by `scripts/docs/generate_reference.py`,
which is why they cannot silently disagree with it: `scripts/check.sh` and
`tests/test_documentation.py` both fail if they are stale. Regenerate with
`uv run python scripts/docs/generate_reference.py`.

| document                                           | generated from                        | what it means                                                                                                 |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [`reference/cli.md`](reference/cli.md)             | the typer app in `src/inmotion/cli/`  | every command, argument, option and default the program actually accepts                                      |
| [`reference/pipelines.md`](reference/pipelines.md) | each pipeline's `RunConfig` dataclass | every pipeline option, so the option list cannot drift from the code                                          |
| [`reference/models.md`](reference/models.md)       | `inmotion.models.registry`            | every registered model: factory, architecture, parameter count, provenance, and what it consumes at its input |
| [`reference/modules.md`](reference/modules.md)     | the source tree                       | every module, its purpose and its public API                                                                  |
| [`reference/datasets.md`](reference/datasets.md)   | `inmotion.datasets`                   | every dataset version: checksum, counts, lineage, label audit                                                 |

Because they are generated, **do not hand-edit them** — the change will be lost on
the next regeneration and the freshness check will fail.

## 4. Checking the claims — audit

| document                                               | what it means                                                                                                                                                                                             |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`audit/AUDIT_CHECKLIST.md`](audit/AUDIT_CHECKLIST.md) | What a reviewer can independently verify, and the command for each row. Written so that nothing requires trusting the authors.                                                                            |
| [`audit/MODEL_CARDS.md`](audit/MODEL_CARDS.md)         | Per-family model cards following the _Model Cards for Model Reporting_ convention: what the model is, what it was trained on, how it was evaluated, its limitations, and what it is **out of scope** for. |

## 5. The record — reports

This is the **history**, not the current interface. `reports/README.md` is the
folder index; the documents fall into three groups.

**Provenance and where things live**

| document                                                     | what it means                                                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`reports/ARTIFACTS.md`](reports/ARTIFACTS.md)               | Where models, logs and results live, which are tracked in git and which are not, and why. Read this before wondering why a checkpoint is "missing". |
| [`reports/MIGRATION.md`](reports/MIGRATION.md)               | Old command → new command, and what has not been reorganised yet.                                                                                   |
| [`reports/REFACTOR_SUMMARY.md`](reports/REFACTOR_SUMMARY.md) | What changed in the reorganisation, with before/after measurements.                                                                                 |

**Review and verification** — who checked what, and against which revision.
These are pinned to a revision on purpose: a verdict is only valid for the bytes
it was taken against.

| document                                                                   | what it means                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`reports/VERIFICATION.md`](reports/VERIFICATION.md)                       | Independent verification of the first closure. Every number was re-run by the verifier; nothing is quoted from the authors.                                                                                                                                                                                      |
| [`reports/VERIFICATION_ROUND2.md`](reports/VERIFICATION_ROUND2.md)         | The round-2 independent verification, after the repairs.                                                                                                                                                                                                                                                         |
| [`reports/CODE_REVIEW_RUST.md`](reports/CODE_REVIEW_RUST.md)               | Round-1 adversarial review of the Rust port and the CPU kernels. Verdict: **needs_revision** — no numerics defect, but packaging and provenance findings.                                                                                                                                                        |
| [`reports/CODE_REVIEW_RUST_ROUND2.md`](reports/CODE_REVIEW_RUST_ROUND2.md) | Round-2 review of the repair of those fourteen findings. Verdict: **pass**.                                                                                                                                                                                                                                      |
| [`reports/CODE_REVIEW_RUST_ROUND3.md`](reports/CODE_REVIEW_RUST_ROUND3.md) | Round-3 final review gate, with a per-item verdict table for every finding across the rounds. Verdict: **pass**. Its §7 records an environment failure that halted verification and the pins that were frozen — kept deliberately, so a reader sees that verification stopped rather than assuming it completed. |

**Measured feasibility and historical design**

| document                                                                 | what it means                                                                                                |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| [`reports/ESP32_FEASIBILITY.md`](reports/ESP32_FEASIBILITY.md)           | Whether the world models fit and run on an ESP32-D0WD-V3, measured on the detected part rather than assumed. |
| [`reports/IMPLEMENTATION_SUMMARY.md`](reports/IMPLEMENTATION_SUMMARY.md) | How the mega-ensemble harness was built (historical).                                                        |

## 6. The published research — papers and results

| directory                          | what it means                                                                                                                                                                                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`isac/`](isac/)                   | The FNWF 2026 paper material.                                                                                                                                                                                                                      |
| [`eucnc/`](eucnc/)                 | The EuCNC/6G Summit submission, **frozen**: LaTeX sources under `eucnc/paper/`, the slide deck under `eucnc/presentation/`, the call for papers, and `eucnc/project_description.md` — the inMotion project described in funder terms (Portuguese). |
| [`icaisf/`](icaisf/)               | Supplementary hyperparameter-optimisation detail for the ICAISF submission: the best configuration per model.                                                                                                                                      |
| [`supplementary/`](supplementary/) | Hyperparameter-optimisation material: `search_spaces.md`, `optimal_hyperparameters.md`, `optuna-summary.md` and a README that frames them.                                                                                                         |

Two cautions. These directories are **frozen for submission**, so editing them
changes what was submitted.

## 7. Non-markdown assets

| path                                         | what it is                                                                                                                                                                                |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eucnc/paper/`                               | The LaTeX source of the EuCNC paper: `main.tex`, `chapters/`, `images/`, `references.bib`, and build output (`.aux`, `.log`, `.pdf`). Build artefacts are committed alongside the source. |
| `eucnc/EuCNC6GS2026_callforPapers_Final.pdf` | The conference call the submission targets.                                                                                                                                               |

## 8. Conventions

- **Generated vs hand-written.** Only `reference/` is generated. Everything else
  is hand-written and therefore can go stale — treat a claim in a hand-written
  document as an assertion about the revision it was written at, not as a
  description of the current tree.
- **History is kept, not tidied.** `reports/` records what was found and fixed,
  including superseded values and retracted diagnoses. Seeing a wrong value in a
  report usually means it is being described **as** wrong; check the surrounding
  sentence before "fixing" it.
- **Pins travel with verdicts.** A review or verification document states the
  revision it judged. If a file it names has since moved, the verdict still holds
  for that revision and must be re-taken for the current one — it does not
  silently extend.
- **Keeping it green.** `uv run python scripts/docs/generate_reference.py --check`
  and `uv run pytest tests/test_documentation.py` fail on stale generated docs and
  on references to paths that no longer exist. Run both after moving or renaming
  anything under `docs/`.
