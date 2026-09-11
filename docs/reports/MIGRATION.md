# Migration notes

## Command mapping

The root scripts still parse `sys.argv` with `argparse`. `inmotion` wraps them so
there is one entry point, and the arguments pass straight through after `--`:

| before | after |
|---|---|
| `uv run python run_exotic.py --model ts_jepa --seed 42` | `inmotion run pipeline exotic -- --model ts_jepa --seed 42` |
| `uv run python run_dl.py --seed 42` | `inmotion run pipeline dl -- --seed 42` |
| `uv run python mega_ensemble.py --data dataset_augmented3.csv` | `inmotion run pipeline mega-ensemble -- --data dataset_augmented3.csv` |
| `uv run python run_classification.py` | `inmotion run pipeline classification` |
| `uv run python export.py --input ... --output ...` | `inmotion data export --input ... --output ...` (native) |
| `uv run python merge_csv.py --input data/ --output dataset.csv` | `inmotion data merge --input data --output dataset.csv` (native) |
| `uv run python clean_dataset.py --data dataset.csv` | `inmotion data clean -- --data dataset.csv` |
| `bash job_exotic.sh` | `sbatch scripts/slurm/job_exotic.sh` |

`inmotion run list` prints every pipeline name.

## Why the delegation is temporary

Rewriting the pipelines to take a config object instead of parsing `argv` is a
behavioural change with real regression risk, and the model-loading work was
higher value. The wrappers are honest about what they are: one entry point, no
behaviour change.

When a pipeline is converted, its `PIPELINES` entry in `src/inmotion/cli/`
becomes a native typer command and the mapping above is dropped.

## What moved

| before | after |
|---|---|
| `wavecom_files/wifi_scan.py` | deleted — byte-identical to `demo/app/wifi_scan.py` and imported by nothing |
| `main.py`, `predictions.py`, `cleanData.py` | `scripts/oneoff/` — unreferenced one-offs |
| `job*.sh` (18 files) | `scripts/{slurm,local,archive}/` |
| `AI_RESULT.md`, `IMPLEMENTATION_SUMMARY.md`, `ROADMAP_TO_90.md`, `research_sources.md`, `.research_plan.md` | `docs/reports/` |

## What is still outstanding

- The pipelines still use `argparse` internally (see above).
- `run_dl.py` (1,635 lines) and `run_exotic.py` (1,562 lines) have not been split
  into modules; the `src/inmotion/classification/` and `src/inmotion/dl/` packages still duplicate
  `DataLoader`, `optimization` and `visualization` responsibilities.
- `src/inmotion/classification/` has not moved under `src/inmotion/`.
- The demo vendors `app/wifi_scan.py` deliberately: its image is
  fastapi+sklearn, and depending on the root package would pull in the
  torch/cu126 stack. That duplication is intentional, not an oversight.
- LaTeX build artifacts under `docs/*/paper/` are untracked but still on disk;
  they are regenerable with `latexmk`.

## Lint status

`src/inmotion/` and `tests/` are clean under the project's ruff configuration.
The legacy `src/inmotion/dl/` package still has 58 findings (mostly unused arguments and
comprehension simplifications) that predate this work and were left alone:
auto-fixing them would touch training code for no functional gain. They are a
follow-up to the `src/inmotion/dl/` → `src/inmotion/models/` move.
