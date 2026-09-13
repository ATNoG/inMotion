# Dataset reference

Generated from `inmotion.datasets`; regenerate the measured fields with
`inmotion datasets manifest --write`. `inmotion datasets verify` re-checks
every file against its recorded checksum.

The datasets are tracked in git, so their history is the version history.
This table adds what git cannot express: checksums, class balance, lineage,
and whether the labels can be trusted.

Canonical training set: **`ds-augmented-v3`**

| id | file | rows | devices | classes | status | label audit |
|---|---|---:|---:|---|---|---|
| `ds-pure` | `data/processed/dataset_only_pure.csv` | 160 | 2 | AA, AB, BA, BB | superseded | known_incorrect |
| `ds-noise` | `data/processed/dataset_only_noise.csv` | 1,196 | 6 | AA, AB, BA, BB | superseded | known_incorrect |
| `ds-augmented-v1` | `data/processed/dataset_augmented.csv` | 10,113 | 8 | AA, AB, BA, BB | superseded | known_incorrect |
| `ds-icaisf` | `data/processed/dataset.icaisf.csv` | 2,251 | 8 | AA, AB, BA, BB | current | unverified |
| `ds-base-v2` | `data/processed/dataset.csv` | 3,511 | 13 | AA, AB, BA, BB | current | unverified |
| `ds-augmented-v2` | `data/processed/dataset_augmented2.csv` | 20,275 | 13 | AA, AB, BA, BB | superseded | unverified |
| `ds-augmented-v3` | `data/processed/dataset_augmented3.csv` | 35,839 | 13 | AA, AB, BA, BB | current | unverified |

## Details

### `ds-pure`

- file: `data/processed/dataset_only_pure.csv`
- rows: 160 (13,962 bytes)
- sha256: `5649b6e3fcda5b3225bd8b9c1b8febb20a137d5043dd863319ab299f3d663a23`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise_label`
- class counts: AA=40, AB=40, BA=40, BB=40
- devices: 2
- has `synthetic` column: False
- first commit: `c221d7f1a3ec` (2026-02-04)
- role: subset; status: **superseded**
- label audit: **known_incorrect**
- note: Early collection. The project owner reports this lineage carries incorrect class labels; kept for provenance only. Do not train on or report from it without re-auditing the labels.
- lineage: `description`=isolated single-device collection, `devices`=2, `successor`=data/processed/dataset_only_noise.csv

### `ds-noise`

- file: `data/processed/dataset_only_noise.csv`
- rows: 1,196 (102,899 bytes)
- sha256: `97c33bb3e07aa1777a2aeedf7346478a7bc765b3bed1860fb21e8fdee7495833`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise_label`
- class counts: AA=298, AB=300, BA=298, BB=300
- devices: 6
- has `synthetic` column: False
- first commit: `c221d7f1a3ec` (2026-02-04)
- role: subset; status: **superseded**
- label audit: **known_incorrect**
- note: Early collection. The project owner reports this lineage carries incorrect class labels; kept for provenance only. Do not train on or report from it without re-auditing the labels.
- lineage: `description`=noisy multi-device collection, `devices`=6, `successor`=8-device pool

### `ds-augmented-v1`

- file: `data/processed/dataset_augmented.csv`
- rows: 10,113 (951,333 bytes)
- sha256: `507973bcd63245b4dd62e3b052a83ac29394e5e06ff9276b7b97d60b399435fa`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise`, `concurrent_noise_path`, `synthetic`
- class counts: AA=2,574, AB=2,490, BA=2,478, BB=2,571
- devices: 8
- has `synthetic` column: True
- first commit: `cd4cb67abe85` (2026-04-25)
- role: augmented; status: **superseded**
- label audit: **known_incorrect**
- note: Early collection. The project owner reports this lineage carries incorrect class labels; kept for provenance only. Do not train on or report from it without re-auditing the labels.
- lineage: `description`=augmentation config 1, `devices`=8, `parent`=8-device pool, `successor`=data/processed/dataset_augmented2.csv

### `ds-icaisf`

- file: `data/processed/dataset.icaisf.csv`
- rows: 2,251 (200,238 bytes)
- sha256: `05f11fb4cba9176d4361302630c02e7c7f09e352ab997b616cb1fa33265a74c9`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise`, `concurrent_noise_path`
- class counts: AA=578, AB=550, BA=546, BB=577
- devices: 8
- has `synthetic` column: False
- first commit: `0f5c0b1c4438` (2026-08-24)
- role: variant; status: **current**
- label audit: **unverified**
- note: ICAISF variant (4 classes AA/AB/BA/BB), 8-device pool. This is the dataset the ISAC paper's ensemble was scored on, so its labels need confirming: if this lineage is the older one.
- lineage: `description`=ICAISF paper variant, `devices`=8

### `ds-base-v2`

- file: `data/processed/dataset.csv`
- rows: 3,511 (312,378 bytes)
- sha256: `39dc03701e7882ac850262500b287e36afab5d7a9657edc4618604b0052dccc1`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise`, `concurrent_noise_path`
- class counts: AA=878, AB=880, BA=876, BB=877
- devices: 13
- has `synthetic` column: False
- first commit: `70d22b298f90` (2025-10-21)
- role: base; status: **current**
- label audit: **unverified**
- note: 13-device pool, collected after the earlier lineage. Source for data/processed/dataset_augmented2.csv and data/processed/dataset_augmented3.csv.
- lineage: `description`=current base dataset, `devices`=13, `successors`=['data/processed/dataset_augmented2.csv', 'data/processed/dataset_augmented3.csv']

### `ds-augmented-v2`

- file: `data/processed/dataset_augmented2.csv`
- rows: 20,275 (1,908,203 bytes)
- sha256: `11fed6d02bd8de70f3c573ece5943097ed5acf48efe78fcd44b0a8f1b11191b7`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise`, `concurrent_noise_path`, `synthetic`
- class counts: AA=5,220, AB=4,930, BA=4,910, BB=5,215
- devices: 13
- has `synthetic` column: True
- first commit: `d465854f25a3` (2026-08-05)
- role: augmented; status: **superseded**
- label audit: **unverified**
- note: Augmentation config 2 over the 13-device pool. Legitimate, but superseded by config 3.
- lineage: `description`=augmentation config 2, `devices`=13, `parent`=data/processed/dataset.csv, `successor`=data/processed/dataset_augmented3.csv

### `ds-augmented-v3`

- file: `data/processed/dataset_augmented3.csv`
- rows: 35,839 (3,370,206 bytes)
- sha256: `82a3c637a70eaf92c431f244b9d51188a80e309ffa2cdad5375f1725752cb0db`
- columns: `mac`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `label`, `noise`, `concurrent_noise_path`, `synthetic`
- class counts: AA=8,962, AB=8,980, BA=8,944, BB=8,953
- devices: 13
- has `synthetic` column: True
- first commit: `d465854f25a3` (2026-08-05)
- role: augmented; status: **current**
- label audit: **unverified**
- note: Canonical training set: the newest augmentation config, built from data/processed/dataset.csv. This is what the trained JEPA/SIGReg checkpoints were trained on, so it is the reference for reproducing their scaling.
- lineage: `description`=augmentation config 3 (latest), `devices`=13, `parent`=data/processed/dataset.csv, `successor`=None

## Schema

| column | meaning |
|---|---|
| `mac` | device identifier. Retained for cross-device analysis and for
  leave-one-device-out evaluation; **not** a model feature. |
| `1`..`10` | RSSI in dBm, one reading per second for 10 seconds. The only
  raw model features. |
| `label` | route taken: `AA`, `AB`, `BA`, `BB`. The prediction target. |
| `noise` / `noise_label` | whether other devices transmitted concurrently. |
| `concurrent_noise_path` | which route the interfering device was taking, when
  known. Used only by the interference analysis, never as a model input. |
| `synthetic` | padding produced by `inmotion data augment` (v2/v3 only). |

## Tiers

| tier | contents | produced by |
|---|---|---|
| `data/raw/` | captures off the access point (Wavecom `.txt`) | field collection |
| `data/interim/` | one CSV per collection session | `inmotion data export` |
| `data/processed/` | merged and augmented training sets |
| | | `inmotion data merge`, `inmotion data augment` |

`inmotion.data.paths.resolve` searches processed, interim and raw in that
order, so a bare filename still resolves.
