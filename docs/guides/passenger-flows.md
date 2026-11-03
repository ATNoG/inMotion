# Synthetic passenger flows: what this is and how to use it

A generator that turns the scraped Aveiro bus timetables into passenger flows —
who boarded, who alighted, who stayed on the bus, who stayed at the stop — and
from those into origin-destination matrices and three ways to look at them.

**The one contract that matters, in one sentence:** the *routes are real* — every
trip, stop, call order and clock comes from the operators' own timetables, and
each run verifies it — while the *riders are generated*, because no operator
records them. Treat the supply as data and the demand as a scenario.

## Quickstart

```bash
set -a && . ./.env && set +a            # optional: CARTO key for real basemap tiles
uv run inmotion bus build               # once: scrape both operators (a few min, needs tesseract)
uv run inmotion bus geocode             # once: place the stops from OpenStreetMap (~1 min cold)
uv run inmotion bus demand --seed 42    # generate a scenario (~3 s, offline)
```

That writes `data/bus/scenarios/s0042-random-bpd12-weekday-weekend/` with the
tables above and three pages. Open any of them:

```bash
xdg-open data/bus/scenarios/s0042-random-bpd12-weekday-weekend/flows_map.html
xdg-open data/bus/scenarios/s0042-random-bpd12-weekday-weekend/flows_cube.html
xdg-open data/bus/scenarios/s0042-random-bpd12-weekday-weekend/flows_replay.html
```

### What that scenario contains, concretely

| | |
|---|---|
| the real network | 1,429 trips, 1,844 stops, 38,804 distinct stop pairs (41,930 pair-days), all from the operators' timetables |
| the generated day | 17,148 riders: 16,419 boarded, 729 never found a bus, 1,252 were refused at a full one |
| busiest stop | Estação (Águeda), 08:15 — 73 waiting, 40 boarded, 33 left behind |
| busiest OD cell | R. Luís G. Carvalho → Av. Dr. Lourenço Peixinho CTT, weekday pm peak — 29 riders over 28 departures |
| the matrix | 11,173 cells; 24 % of the pairs carry anyone; peak load factor 1.00 |
| the cube | 2,646 rides over the busiest 60 stops, 312 of them changing hour between boarding and alighting |
| the replay | 106 routes playable; 604 of the 1,429 trips (the rest call at a stop with no coordinates, or run backwards in the scrape) |
| verification | every trip, stop visit and OD row traced to the scrape; `problems: []` in `report.json` |

### A day busy enough to watch

```bash
uv run inmotion bus demand --seed 42 --boardings-per-departure 40 --out data/bus/scenarios/demo-dense
xdg-open data/bus/scenarios/demo-dense/flows_replay.html
```

Same network, four times the riders: the peak bus fills 40/40 and five departures
leave people behind, so the replay shows `N left` badges instead of empty stops.

## What you get, per scenario

Every run writes one directory with these tables (all CSV, read them with
`schema_overrides={"line": pl.String}` — AveiroBus numbers two of its lines
`azul` and `verde`):

| file | one row per | answers |
|---|---|---|
| `flows.csv` | trip and stop | arrivals, boarded, alighted, load before/after, load factor, riders left behind |
| `stop_flow.csv` | stop and hour | the queue: who arrived, who boarded, who was still waiting |
| `od_flow.csv` | trip, origin and destination | the ride actually taken, with its boarding and alighting clock |
| `od_demand.csv` | stop pair | the demand before any bus could refuse it |
| `od_passengers.csv` | operator, service day, band, stop pair | the passenger OD matrix, keyed like the supply side's `od_matrix.csv` |
| `od_passengers_timeline.csv` | the same, per hour | the dynamic OD matrix (slices sum to the static one) |
| `od_passengers_wide_*.csv` | origin × destination grid | the same as a dense grid, per operator and service day |
| `instants.csv` | stop visit and instant | the `AA`/`BB`/`BA`/`AB` label grid at 10 s, the vocabulary the WiFi work uses |
| `scenario.json` | the run | every parameter, the sha256 of the inputs it read |
| `report.json` | the run | counts, invariants checked, provenance, warnings |

And three self-contained pages, each openable by double-clicking:

| page | what it shows |
|---|---|
| `flows_map.html` | the network on a map: stops sized by boardings, OD lines by riders, filters and a clock |
| `flows_cube.html` | the same OD relation as a space-time cube: origin × destination × hour, entries vs exits |
| `flows_replay.html` | one route's day played back: buses moving on the timetable, people waiting as dots, boarding, alighting, who was left behind |

A batch adds `index.csv`: one row per scenario, which is the table to read when
comparing scenarios.

## Every command

```bash
uv run inmotion bus build      # once: scrape + parse the operators into data/bus/processed
uv run inmotion bus geocode    # once: place the stops from OpenStreetMap (cached, ~1 min cold)
uv run inmotion bus demand     # generate one scenario (all of the above, offline)
uv run inmotion bus demand --help    # the authoritative flag list
```

The flags that change what you get:

| flag | effect |
|---|---|
| `-n/--scenarios N` | sample and generate N scenarios into one batch (with `index.csv`) |
| `--seed S` | the batch's seed; scenario *i* uses `S+i`. Same seed + same inputs = identical tables |
| `--boardings-per-departure X` | the demand dial when no daily total is given (default 12) |
| `--daily-boardings N` | total riders for the day, split across the network instead |
| `--fleet "name:capacity:count,..."` | the fleet: `minibus:20:2,standard:40:8,articulated:70:3` |
| `--fleet-policy random\|cycle\|demand` | how bus sizes reach trips: sampled, round-robin, or biggest to busiest lines |
| `--alpha`, `--beta` | stop attraction exponent and journey-time decay (the gravity shape) |
| `--service weekday\|weekend\|both` | which days to generate |
| `--stops FILE` | per-stop CSV with `stop_id,weight,category,lat,lon`: weight overrides and coordinates that beat any match |
| `--step N` | seconds between ground-truth instants (default 10) |
| `--scenario-dir/--out DIR` | where to write (a single run writes into DIR; a batch makes one subdirectory per scenario) |
| `--no-instants`, `--no-map`, `--no-cube`, `--no-replay` | skip the expensive outputs |
| `--cube-stops N` | how many stops the cube's lattice holds (default 60) |
| `--basemap auto\|carto-dark\|carto-light\|network\|osm\|none` | what the pages draw under the data |
| `--refresh` (on `geocode`) | re-fetch the OpenStreetMap index instead of using the cache |

## Recipes

**One scenario, the one every result is documented against**

```bash
set -a && . ./.env && set +a          # the CARTO key, if you want a real basemap
uv run inmotion bus demand --seed 42
```

**A batch to compare hypotheses** — fleet policy is the usual axis:

```bash
uv run inmotion bus demand -n 8 --seed 42
column -s, -t data/bus/scenarios/index.csv | less -S
```

**An explicit grid** (cross product, not a zip) — `sweep.json`:

```json
{"beta": [0.02, 0.06, 0.12], "fleet_policy": ["random", "demand"]}
```

```bash
uv run inmotion bus demand --sweep sweep.json
```

**A day dense enough to watch** — the default dial spreads ~16 000 riders over
41 000 stop pairs, so any single route looks thin:

```bash
uv run inmotion bus demand --seed 42 --boardings-per-departure 40 --out data/bus/scenarios/demo-dense
```

**One line, no code** — read the tables:

```bash
uv run python -c "
import polars as pl
d = 'data/bus/scenarios/REPLACE-ME'
read = lambda n: pl.read_csv(f'{d}/{n}', schema_overrides={'line': pl.String})
print(read('od_passengers.csv').sort('passengers', descending=False).head())
"
```

## Working in a group

**What to share.** Commit small, and regenerate large:

* `data/bus/processed/` — the scraped tables and the coordinate tables (~200 KB).
  These make the generator runnable offline for everyone.
* A scenario's `scenario.json` and `report.json` (a few KB each): they are the
  record — every dial, the inputs' checksums, every invariant that was checked.
* `index.csv` from a batch: the comparison table.

**What not to share.** A scenario directory is 40–170 MB per day of output and is
regenerated from its `scenario.json` in seconds, so it does not belong in git.
The map pages also embed the CARTO key in their tile URL — that key belongs in
`.env` only, and a scenario built with it should not be committed. The credentials
guard in `tests/test_no_secrets.py` knows a CARTO key's shape and scans the source
tree for it.

**How a colleague reproduces your scenario exactly.** Send `scenario.json`; they
run the same command with the same `--seed`. Same seed and same inputs produce
byte-identical tables — the suite asserts it. If the supply tables changed in
between, `scenario.json`'s input checksums show it, and the counts in
`report.json` will differ from theirs — the seed alone does not make two
different timetables comparable.

**Naming.** A scenario's directory name is derived from its dials
(`s0042-demand-bpd12-weekday`), so two runs of the same thing collide and
different things never do. Use `--seed` to keep a family of runs together.

**Comparing.** Never compare two scenarios by eye across files: read `index.csv`
(label, seed, policy, demand, boardings, denied, never boarded, peak load factor,
problems). `problems` must be `0`; anything else means the run failed its own
invariants and the numbers are not usable.

**Quoting.** Every claim from this data needs its supply/demand split stated:
"on the operator timetable of <date>, with a generated day of N riders …". The
provenance check in each run states what was verified:

```
supply verified: 1,429 trips, 18,792 stop visits and 15,153 OD rows all trace to
the scraped timetable (the demand itself is generated)
```

## What it is not

* **Not a ridership estimate.** The dials are conventional, not calibrated: no
  counts were available to fit them. A scenario is a plausible day, not a
  measured one.
* **No transfers.** Only the direct pairs the timetable serves; a journey needing
  a change is outside the model.
* **One service day at a time.** Weekday and weekend are separate days, and a
  queue never boards a bus from the other service.
* **Not every route is fully mappable.** Coordinates resolved for 1,389 of 1,844
  stops (75 %); the map shows those, and the replay only plays trips whose every
  stop is placed.
* **Buses fill exactly.** Boarding is capped at the free seats with no no-shows,
  so a saturated departure sits at a load factor of 1.00.

## Where the details live

* `docs/reference/cli.md` — every command and option, generated from the code.
* `docs/reference/modules.md` — every module and its public API.
* `AI_RESULT.md` section 2 — what was produced from the scraped network, the
  coordinate tiers, the invariants, and the known limitations.
* `src/inmotion/bus/demand/` — the package; each module's docstring states the
  rule it enforces and why.
