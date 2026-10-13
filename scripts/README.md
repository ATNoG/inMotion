# Scripts

Run commands from the **repository root** — every script here assumes that
working directory (they use paths like `dataset.csv` and `run_exotic.py`).

## `slurm/` — cluster submissions

Have `#SBATCH` headers; submit with `sbatch scripts/slurm/<name>.sh`.

| script | purpose |
|---|---|
| `job.sh` | baseline DL run |
| `job_normal.sh`, `job_noise.sh`, `job_pure.sh` | DL run per dataset variant |
| `job_aug.sh` | augmented DL run, seeds 42/3/5 |
| `job_exotic.sh` | LeJEPA + SIGReg (exotic pipeline) |
| `job_full_train.sh` | full retrain at HPO-best configs |
| `job_master_train_v2.sh` | rebuild all ensemble members (current) |
| `job_multi_seed.sh` | multi-seed runs for ensemble diversity |
| `job_interfer.sh` | cross-route interference plots |

The `#SBATCH --output=` paths are relative and were left alone to avoid breaking
existing submissions; create `logs/` first if a job fails to start.

## `local/` — single-machine overnight runs

No `#SBATCH`; launched with `nohup bash scripts/local/<name>.sh & disown`.

`job_t_jepa.sh`, `job_ts_jepa.sh`, `job_cf_jepa.sh`, `job_lejepa_rdm.sh`,
`job_rdm_sequential.sh`, `job_multi_seed_ff.sh`.

## `archive/` — superseded, kept for provenance

| script | superseded by |
|---|---|
| `job_master_train.sh` | `slurm/job_master_train_v2.sh` (v2 header documents the v1 bugs) |
| `job_aug_2.sh` | `slurm/job_aug.sh` (re-runs a single seed already covered) |

## `oneoff/` — one analysis, kept so its figures can be re-derived

Not part of any pipeline: each reads what a build already wrote and writes
figures, diagnostics or its own derived tables. Nothing here is imported by the
package, and no script here writes a filename the package also writes.

| script | what it does |
|---|---|
| `cleanData.py`, `main.py`, `predictions.py` | early dataset cleaning and prediction dumps, from before the CLI |
| `check_triangularity_2trips.py` | the two-trip case that settles whether an OD matrix is triangular |
| `diagnose_triangularity.py` | classifies why a built matrix is not triangular, line-direction by line-direction |
| `make_route_sorted_matrices.py` | re-orders the wide OD matrices by travel order instead of by stop id, and the stop-order tables that go with them |
| `make_tier_a.py`, `make_tier_b.py` | the tier-A and tier-B bus figures (departure bands, headway, reachability, next-hop) |
| `make_plots.py` | the two overview bus figures |

The bus tables these read come from `uv run inmotion bus build`. The figures go
to `plots/bus/`; the route-ordered matrices and stop-order tables land beside
that build in `data/bus/processed/` under their own `_sorted` / `od_stop_order`
names, so a rebuild neither produces nor overwrites them.

## Credentials

Training scripts read `WANDB_API_KEY` from the environment and fail fast when it
is missing. Copy `env.example` to `.env` and source it. A key used to be
hardcoded in these files, so it is in the git history and must be rotated.
