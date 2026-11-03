# inMotion — results

This is the project's single results document. It currently holds the
origin-destination work for the Aveiro bus networks; the classification results
described in [`AGENTS.md`](AGENTS.md) are appended here as they are produced
rather than in separate files.

---

# 1. Aveiro bus networks and origin-destination matrices

## 1.1 What was asked

Scrape `aveirobus.pt` and `busway-cira.pt` — driving the sites with the CUA
(computer-use) driver — to obtain **every trip with every stop across all
schedules**, and derive OD matrices from that. The data is expected to be big.

## 1.2 What the two sites actually publish

Neither operator publishes GTFS, an API, or any machine-readable feed, so the
timetables only exist as the pages themselves:

| operator | network | published as |
|---|---|---|
| **BusWay (CIRA)** | intermunicipal and inter-regional, 107 lines | one page per line and direction; each service type is an HTML table of **stops against trips**, so one table already contains every stop of every departure |
| **AveiroBus** | the city network, 14 lines | one PDF per line; the timetable inside is a **single embedded bitmap** (~3400×2200 px, ≈430 dpi). `pdftotext` returns only the page number, and two of the fourteen (lines 11 and 13) carry a real text layer instead |

A BusWay page is therefore the whole timetable for that departure direction,
but only the *outbound* direction is linked from the municipality pages; the
complete set comes from the operator's own sitemap, which lists all 197 pages
by slug.

## 1.3 How the CUA driver was used

The CUA driver (`computer-use-linux`) drove a **dedicated Chrome instance** —
its own profile, so the session in the user's own browser was never touched —
through both sites, and captured the rendering of the pages the parser reads:

| evidence | what it shows |
|---|---|
| [`data/bus/cua/busway-line-16-ida.jpg`](data/bus/cua/busway-line-16-ida.jpg) | `busway-cira.pt/linhas/16-ida/` as rendered: the season row (`A A A A PE A A A PE …`) and the highlighted `07:58* / 08:48* / 14:46*` cells |
| [`data/bus/cua/aveirobus-line-1.jpg`](data/bus/cua/aveirobus-line-1.jpg) | the AveiroBus line 1 PDF opened in Chrome: both timetable blocks, the lavender term-only column, the footnote |

Screenshot capture is bounded (≤1920 px, and Chrome's PDF viewer scales the
page down), which is exactly why the numbers are **not** read from the screen:
OCR of a screenshot of a bitmap timetable is a lossy second-generation copy.
Instead the browser is used to establish *which* pages exist and to verify the
result, and the bytes the browser itself receives are then read exactly (the
HTML table, or the embedded bitmap at its native resolution).

Each CUA screenshot was checked against the parsed rows. Line 16, outbound,
third weekday departure, and the whole of AveiroBus line 1, outbound block,
match to the minute (see §1.8).

## 1.4 Extraction

Both parsers live in [`src/inmotion/bus/`](src/inmotion/bus) and are run by
`inmotion bus build`.

**BusWay** ([`busway.py`](src/inmotion/bus/busway.py)) parses the HTML table
directly. One table is one service type; one column is one trip; one row is one
stop, so a cell is a single time — or a dash, meaning the trip does not call
there. The season row above the columns (`A` = *Anual*, `PE` = *Período
Escolar*) attaches to every trip in that column, and two markup variants are in
use across the site (`epoca_1_20`, `epoca_M_Sever_Vouga`), so classes are
matched by prefix. A starred, highlighted cell records both a footnote (a
connection to another line) and the connection itself.

**AveiroBus** ([`aveirobus.py`](src/inmotion/bus/aveirobus.py)) lifts the
embedded bitmap out at native resolution and recovers the **grid** from the
pixels before reading anything:

1. header bands are found by **colour saturation**, not by one hard-coded
   colour, because every line uses a different one (red on line 1, blue on
   `azul`, green on `verde`, magenta on line 4, and on line 12 the alternating
   row shading is itself saturated). A band must also carry a service caption,
   which is what separates a header from line 12's shading;
2. a single strip may hold several captions side by side (`DIAS ÚTEIS … FIM DE
   SEMANA E FERIADOS`), so each strip is split into one band per caption and the
   captions partition the strip's columns between them;
3. rows come from the times, the dashes **and the stop names**, so a stop whose
   time the OCR missed is still a stop of the line;
4. every cell is read from **two independent OCR passes** (different
   segmentation, scale and binarisation). A cell they disagree on is re-read
   from a tight crop of itself; so is every cell left empty, which is what turns
   "not served" from a guess into a reading;
5. a tinted column is the operator saying the departure runs under a restricted
   calendar, so the tint is read off the pixels and kept;
6. `23:58` followed by `00:02` is one trip crossing midnight, not a trip running
   backwards, so the column is put on a continuous scale.

The two text-layer PDFs are parsed from their real text with the same grid
reconstruction, no OCR involved.

## 1.5 What was produced

`inmotion bus build` writes `data/bus/processed/`:

| file | rows | contents |
|---|---|---|
| `stop_times.csv` | 24,437 | the tidy timetable: one row per trip and stop, with `time`, `minutes`, `served` and markers |
| `trips.csv` | 1,429 | one row per trip: line, direction, service, season, endpoints, span, term-only flag, column tint |
| `stops.csv` | 1,870 | the stop dictionary per operator, with how many trips call and which lines |
| `od_pairs.csv` | 97,282 | the OD relation per line, direction, service and departure band: direct departures per stop pair, first and last departure, min/median/max journey time |
| `od_matrix.csv` | 86,834 | the matrix, aggregated over lines: operator, service, departure band and stop pair against direct departures |
| `od_timeline.csv` | 131,728 | the same matrix sliced by time interval instead of band: operator, service, slice and stop pair against direct departures |
| `od_matrix_<operator>.csv` | — | the same, split by company, with the `operator` column dropped |
| `od_matrix_wide_<operator>_weekday.csv` | — | the same as a dense origins × destinations grid, per company |
| `od_matrix_wide_<operator>_weekend.csv` | — | as above, for the weekend service |
| `build.json` | — | the counts, the OCR repair count and every warning |

Per `data/bus/interim/` there is one `*_stop_times.csv` per operator, so a
parser can be checked on its own.

Totals, from that build:

```
busway pages   197        trips          1429
aveirobus pdfs 14         stop times     24437
od pairs       97282      od cells       86834
timeline cells 131728     ocr repairs    103
midnight rolls 3
```

| | lines | stops | trips (weekday / weekend) | OD pairs |
|---|---|---|---|---|
| BusWay | 107 | 1,673 | 789 / 161 | 78,704 |
| AveiroBus | 14 | 197 | 304 / 175 | 18,578 |

283 trips carry the `PE` (term-time) season and 12 are tinted as running under a
restricted calendar.

## 1.6 What the OD matrices mean

Nobody records how many people travelled, so what a timetable can produce is a
**service** OD matrix: for each ordered pair of stops, how many **direct** trips
connect them, how long that takes, and when the first and last one leaves. It is
the supply an accessibility or headway study needs, and it is not a passenger
count.

Each row of `od_matrix.csv` is one `(operator, service_group, departure_band,
origin, destination)` combination, so a pair that is served in both bands by
both a weekday and a weekend service appears in several rows; `trips` is the
number of departures, and `min/median/max_run_min` the spread of journey times
across them. Departure bands are `early` (<07:00), `am_peak`, `midday`,
`pm_peak`, `evening` (<24:00) and `night` (past midnight, which Portuguese
timetables do print).

**Split by company, and by shape.** The two networks share no stop ids, so a
single grid over their union is two disjoint blocks in 1,819 columns; the
matrices are therefore written per operator. Two shapes are provided:

* **wide** (`od_matrix_wide_<operator>_<service>.csv`) — the matrix as a grid.
  One **row per origin stop**, one **column per destination stop**, and the cell
  is **how many direct trips** that operator runs between the two, for that
  service. A zero means no direct trip, which is not the same as unreachable:
  it may still be reachable with a change, and the pass-through trips are the
  reason the grid is not simply a triangle of ones.
* **long** (`od_matrix_<operator>.csv`, or `od_matrix.csv` for both) — the same
  numbers as one row per cell, which is what a pivot or a join wants.

Long form, column by column:

| column | meaning |
|---|---|
| `service_group` | `weekday` or `weekend` (the operators' own labels are `dias úteis`, `sábados`, `fins de semana e feriados`; AveiroBus only distinguishes weekday from weekend) |
| `departure_band` | when the trip leaves its origin: `early`, `am_peak`, `midday`, `pm_peak`, `evening`, `night` |
| `origin_id`, `origin` | the stop the passenger boards at |
| `destination_id`, `destination` | the stop they get off at |
| `trips` | **direct departures** covering that pair — the cell of the matrix. A circular route that passes the same two stops twice counts twice, because a rider can take it twice; a pair printed twice in one minute by the operator counts once, because it is one call |
| `lines` | which lines provide it |
| `first_minutes`, `last_minutes` | the earliest and latest departure time of day |
| `min_run_min`, `median_run_min`, `max_run_min` | journey time by the fastest, median and slowest of those departures |

So one row reads: *on a weekday during the morning peak, AveiroBus runs 24
direct departures from R. Luís G. Carvalho to Caçadores 10 — Misericórdia, the
first leaving at 07:10, the median one taking 6 minutes.*

Reading the tables back needs `dtype` on the line column, because AveiroBus
numbers its lines `1`…`13` while BusWay's are numeric strings and the site also
publishes `azul` and `verde`:

```python
import pandas as pd
trips = pd.read_csv("data/bus/processed/trips.csv", dtype={"line": str})
od = pd.read_csv("data/bus/processed/od_matrix.csv")
od[od.service_group == "weekday"].nlargest(10, "trips")
```

The busiest single pairs in the data are inner-city Aveiro hops — *R. Luís G.
Carvalho → Caçadores 10 — Misericórdia* is served 109 times on weekdays with a
median journey of 5 minutes — and the busiest lines are BusWay 36 (105 trips)
and 18 (72), then AveiroBus 4 (53) and 11 (46).

## 1.7 OD matrices by time interval

The matrix above aggregates a whole service day. `od_timeline.csv` is the same
relation with time kept in it: one row per operator, service group, **interval**
and stop pair, carrying the same measures, so `M_ij(t)` — the direct departures
from `i` to `j` that left inside interval `t` — can be read off it.

The interval is stamped on the departure from the row's own origin, the instant
the departure bands already use. Two consequences worth naming: a trip is never
split between slices, so a 23:50 departure that arrives after midnight belongs to
`23:00-24:00` in full, and a rider who boards mid-trip is counted in the slice
they left in, not the one the vehicle started in. The slices are 60 minutes
aligned to 00:00 and they keep counting past 24:00, where the timetables still
print.

23 slices carry flow: the eighteen hours from 06:00 to 24:00, an hour at 00:00
and one at 04:00, the after-midnight `24:00-25:00` (three weekend trips on
AveiroBus 4 that really do leave after midnight), and `34:00-35:00` and
`40:00-41:00`, which hold nothing but the misread line-10 cells §1.10 lists. A
slicing that shows you your own bad cells is doing its job. The busiest slice is
not the same hour for both services:

| operator | service | departures | busiest slice | departures there |
|---|---|---|---|---|
| BusWay | weekday | 102,623 | 07:00-08:00 | 17,392 |
| BusWay | weekend | 4,247 | 08:00-09:00 | 762 |
| AveiroBus | weekday | 32,498 | 13:00-14:00 | 3,058 |
| AveiroBus | weekend | 18,816 | 21:00-22:00 | 1,372 |

Because the windows partition the service day, the slices add back up to the
static matrix cell by cell: 158,184 direct departures either way, checked on
every build by `inmotion.bus.od.validate.validate_timeline`. That check is what
found the line-107 duplicate in §1.9 — as bands the two copies sat inside one
row, and only as hours did the totals stop matching.

A different window is one command, and costs nothing, because the slices are cut
from the processed table rather than from the sites:

```bash
uv run inmotion bus timeline --window 30 --step 30 --out plots/od-30min
uv run inmotion bus timeline --wide             # + one dense grid per slice
```

`--step` shorter than the window overlaps the slices, so a trip is then counted
by every slice containing its departure and the slices must not be summed; a
longer step samples the day instead, leaving the gaps unread. `--wide` writes one
grid per operator, service and slice, `od_timeline_wide_<operator>_<service>_<HHMM>.csv`,
all on that operator's whole-day stop axes, so the series is stackable and a quiet
interval is zeros rather than a smaller matrix.

## 1.8 Checking two routes by eye

`inmotion bus plot` draws one figure per operator, in `plots/bus/`. The pair is
not chosen by hand: [`similar_pairs`](src/inmotion/bus/viz.py) ranks every pair of
an operator's routes by how much of their stop list they share, and the top pair
is drawn. Each figure carries three views of that pair:

* **the OD grid of each route**, in that route's own travel order. Read
  correctly it is upper-triangular — a trip never runs backwards and a stop is
  never its own destination — and the served cells are exactly the pairs the
  route can carry someone between.
* **a route ladder for each**, stops in travel order against the minutes since
  that route's own first stop, with every shared stop joined by a connector.
  A mis-parsed route shows up as a jagged ladder and connectors crossing at
  random.
* **a printed agreement number**: the Spearman correlation between the two
  routes' positions of their shared stops. `−1` means the second route is the
  first reversed, `+1` means the two run the same corridor in the same order.

What the data chose:

| operator | pair | shared stops | Jaccard | agreement |
|---|---|---|---|---|
| BusWay | line 103, ida vs volta | 32 of 32 | 1.00 | **−0.93** → mirror order |
| BusWay | line 13, ida vs volta | 7 of 7 | 1.00 | −1.00 → mirror order |
| BusWay | line 42, ida vs volta | 28 of 29 | 0.97 | −1.00 → mirror order |
| AveiroBus | line 1 b1 vs line 5 b1 | 8 of 18 | 0.44 | **+1.00** → same order |

That is the extraction validating itself on two different axes. A line's two
directions should be the same 32 stops in reverse, and they are: the ladders in
[`plots/bus/route_pair_busway.png`](plots/bus/route_pair_busway.png) are exact
mirrors of one another and the connectors run near-parallel, which is what
`−0.93` says. Two *different* lines that share a corridor should visit the
shared stops in the same order, and AveiroBus 1 and 5 do — 8 stops of the
Aveiro–Esgueira corridor, `+1.00` in
[`plots/bus/route_pair_aveirobus.png`](plots/bus/route_pair_aveirobus.png).

The figures also show what the matrix does *not* contain: BusWay 103's grid has
462 of 1,024 cells served on the outward run and 399 on the return, because
several trips are short-workings that turn back before the terminus — the dark
sub-block at `Escola Secundária` is the school-day variant. A triangular grid of
ones would have been the sign of a parser that had collapsed every trip onto the
line's full stop list.

## 1.9 Verification

**Against the live sites, through the CUA driver.** Line 16 outbound, third
weekday column, from the screenshot: `R. Nova de Santiago B 06:40`, `Hospital /
Universidade B 06:42`, `Estação 06:47`, `Areal (Angeja) 07:09`, and the season
row `A A A A PE A A A PE …`. The parsed row is identical, and the page yields
20 `A` and 2 `PE` trips. AveiroBus line 1's outbound block parses to 14 stops in
every one of its 18 trips, the first from `06:43` to `07:29`, with trip 3 — and
only trip 3 — tinted `#a890c0`, matching the lavender column in the screenshot.

**Invariants** (`inmotion.bus.od.validate`, run by every build): no stop time
outside 00:00–48:00, no printed trip whose times run backwards, no OD pair
claiming zero trips, and no negative journey time. These checks are what found
the two most serious defects in this work:

* BusWay pages carry one table per service type and each numbers its trips from
  1, so indexing the page-wide trip list made the second table append its
  departures onto the first table's trips — doubling call lists and interleaving
  two services into one trip. **103 trips ran backwards; the fix left 3.**
* BusWay's own `15-volta` page carries a copy-pasted `15 – IDA` heading, so
  taking the heading as the trip's direction merged two pages' trips under one
  id. Trip ids now come from the URL slug, and the heading only supplies the
  caption and a fallback for a slug with no direction (the summer line's
  `l35`).

* The line 107 page prints `Rio Tinto Sul` **twice at the same minute**, so the
  relation counted one call as two, and every pair through that stop twice over.
  It was found by the interval cross-check in §1.7 and not by eye: two rows in
  two trips, 104 pair rows in all, invisible in any total. The tidy table still
  records both rows, because it says what the page says; the relation collapses
  a stop printed twice in a row at the same minute into one call.

**Cross-checks on the OCR.** Two independent passes per bitmap, a third reading
from a tight crop of the cell whenever they disagree, and a fourth whenever the
result would break the "a trip's times never go backwards" invariant. 103 cells
were settled that way across 24,437 stop times.

## 1.10 Known limitations

* **Six trips still run backwards**, in seven cells where no OCR pass produced a
  usable value. Two are a leading `1` lost outright: AveiroBus `10:b1:weekday:9`
  (`14:45` then `04:52`) and `8:b1:weekday:10` (`17:06` then `07:15`). The rest:
  `10:b1:weekday:2` (`08:27` then `08:07`), `10:b2:weekday:4` (`10:20` then
  `00:21`, and `40:41` for a value that cannot be 40:41), `5:b1:weekday:11`
  (`21:37` then `20:59`), and BusWay `17-volta:weekday:6` (`17:15` then
  `17:02`). They affect 17 of 97,282 OD pairs (0.017%). They are
  kept rather than deleted — a wrong time is visible, a deleted one is not —
  and every one is listed in `data/bus/processed/build.json`.
* **The matrices are supply, not demand.** No passenger counts, boarding
  counts or vehicle capacities exist in the sources.
* **No coordinates.** Neither site publishes stop positions in machine-readable
  form, so the matrices are keyed by stop name, not by a point. Stop identity is
  per operator and normalised (`Qtª`, `Qt?` and `Qta` all key to `qt`), but the
  same physical stop appearing in both networks is *not* joined.
* **A snapshot, not a calendar.** A timetable also carries validity dates
  (`Entrada em vigor`, and BusWay's `PE` season) which are recorded per trip but
  not expanded into a date-by-date calendar.
* **Line 12's three tint colours** (school-calendar variants, explained by a
  legend printed inside the bitmap) are captured as a tint and a `school_only`
  flag, but the legend text itself is not read, so which variant is which is not
  recorded. Line 11's page also yields one caption-less band at the foot of the
  page, reported as a warning.
* **Times are indicative.** BusWay says so on every page: *"Tempos de passagem
  meramente indicativos e não vinculativos"*.

## 1.11 Reproducing

```bash
uv run inmotion bus fetch     # data/bus/raw + manifest.json (checksums)
uv run inmotion bus build     # data/bus/{interim,processed} + the hourly timeline
uv run inmotion bus timeline --window 30 --step 30 --out plots/od-30min
uv run inmotion bus plot      # plots/bus/route_pair_<operator>.png
uv run pytest tests/test_bus.py -q
```

The ~109 MB of operator HTML and PDFs under `data/bus/raw/` is gitignored;
`data/bus/raw/manifest.json` records the URL, size and sha256 of every artifact,
so the scrape replays against the live sites and a page that has changed is
detected rather than absorbed. OCR needs the `tesseract` binary with the `por`
language pack *and* the `tsv` config that ships beside it in the tessdata
directory (`TESSDATA_PREFIX` may point at a directory holding both); without
them `bus build` refuses instead of writing BusWay-only tables. The Python side
is `pymupdf`, `pillow` and `numpy`.

The readings above come from the official `tessdata_fast` `por` model, whose
file has not changed upstream since 2017. The counts are those of the parser as
it stands: an earlier run of the same corpus with the pre-fix aggregation read
24,416 stop times and 94 repairs, and filed every trip of a line under one
departure band.

# 2. Synthetic passenger flows and origin-destination matrices

## 2.1 What was asked

The two networks publish supply and nothing else: a timetable says where the
buses go and when, and no operator records who was on them. What was asked for
was a *generator*: given the bus sizes, the stops, and the trips with their
schedules, produce passenger flows — entries, exits, who stayed on the bus, who
stayed at the stop — and from those the OD matrices. Coordinates were the one
thing the scraped data does not carry, so they had to come from somewhere else,
which is where OpenStreetMap comes in.

## 2.2 How the demand is generated

Nobody recorded how many people travelled, so the demand is invented, and the
model is deliberately the smallest one with the right shape:

* **Attraction** `A_i = w_i · (1 + lines_i)^α` — what a stop draws, from its own
  service level, optionally overridden per stop by an attribute file (a
  hospital, a campus, a stop whose weight is known).
* **Deterrence** `e^(−β·t_ij)` — how sharply a destination is avoided as the ride
  gets longer, on the quickest direct ride between the two stops.
* **The day** `T_ij ∝ A_i · A_j · e^(−β·t_ij)`, normalised to the day's boarding
  target and **sampled**, not rounded: each rider's pair is drawn from the
  weights. Rounding the expected value instead gives every one of the 41,930
  stop pairs a third of a rider and empties all but a few hundred of them; a
  real OD matrix is sparse with a long tail, and drawing the riders is what
  produces that. The draw is a multinomial, so the total is still exact.
* **The buses.** Each pair's riders are spread over the departures that serve it,
  weighted by the departure band, so demand follows the commutes rather than the
  timetable's even spread. Every trip gets a bus size — by the scenario's fleet
  policy: `random` (a seeded draw over the fleet mix), `cycle` (in departure
  order) or `demand` (the biggest vehicles to the busiest lines).
* **The queue.** Buses are then run over their stops in the order they *arrive at
  each stop*, and at each one: riders whose destination is this stop get off,
  the riders who turned up for this departure join the queue, and as many as
  there are free seats get on, split across destinations by largest remainder
  when the bus fills. Whoever is left stays at the stop and is picked up by the
  next bus for their destination; what is left when the day's last bus has gone
  is reported as *never boarded* rather than dropped.

The four quantities that were asked for fall out of that one model:
`arrivals`/`boarded` are the entries, `alighted` the exits, `onboard_before` and
`onboard_after` who stayed on the bus, and `waiting_start`/`waiting_end` the
queue at the stop. Since the rest of inMotion labels 10-second WiFi captures as
`AA` (stayed inside the bus), `BB` (stayed at the stop), `BA` (boarded) and `AB`
(alighted), `instants.csv` writes the same day in that vocabulary, one row per
stop visit and instant.

The dials are conventional, not calibrated: `α = 1`, `β = 0.06` per minute, and
12 boardings per departure put the scraped network at ~17,000 riders a day,
which is the order of a small regional operator. Every number reproduces from
`scenario.json`.

## 2.3 Stop coordinates

`inmotion bus geocode` makes one Overpass request for every named bus stop of
the Aveiro district (5,592 of them), caches it under `data/bus/raw/osm`, and then
matches names locally — no per-stop web traffic, and a scenario run never
touches the network at all. Each tier is recorded per stop, so an analysis can
say what it trusts:

| source | stops | what it means |
|---|---|---|
| `tokens` | 884 | one name's words are contained in the other's, either way ("Barrocas A" against "Barrocas") |
| `exact` | 467 | the canonical name matches an OSM stop exactly |
| `fuzzy` | 25 | a close spelling, tie-broken by distance to the line's other stops |
| `expanded` | 11 | matched after spelling out `R.`, `Av.`, `Qtª` and the rest |
| `sibling` | 2 | the other direction's stop of the same place resolved |
| `none` | 455 | nothing matched, left blank on purpose |
| **placed** | **1,389 of 1,844** | **75.3 %** |

A stop with `source = none` is left empty and listed in `stops_unresolved.csv`.
Guessing a position would put a phantom stop on the map and, worse, a plausible
one in the middle of an analysis, so the gap is reported instead.

The remaining 455 are street-named stops — `R. Clube dos Galitos A`, `Av. da
Força Aérea B`, `Forca Hospital A` — for which OSM either has no named bus stop
or names it after something else. Asking Nominatim about them does not help: on
a 40-stop sample of the unresolved list, 18 lookups returned a place and **none
of them was a bus stop** (six were residential streets, two hamlets, and the
rest houses, museums and government offices), which is why `--web` is opt-in and
documented as adding nothing here rather than run by default. Broadening the
Overpass selectors adds little either: `public_transport=platform` without
`bus=yes` is 151 more objects than with it, against 5,735 with.

Stops within 60 m of each other are then grouped into 994 *places*
(`stop_clusters.csv`), so the same corner served by both operators reads as one
place on the map. The grouping is single-linkage, so a dense street of stops can
chain into one large cluster; the 60 m radius is the knob.

## 2.4 What was produced

`inmotion bus demand -n 1 --seed 42` writes one directory per scenario under
`data/bus/scenarios/`. For `s0042-random-bpd12-weekday-weekend`:

| file | rows | contents |
|---|---|---|
| `flows.csv` | 18,792 | per trip and stop: arrivals, boarded, alighted, the load before and after, the load factor, and the riders left behind |
| `stop_flow.csv` | 21,504 | per stop and hour: the queue — arrived, boarded, refused, and the queue at each end of the hour |
| `od_flow.csv` | 15,153 | per trip, boarding stop and destination: the ride actually taken |
| `od_demand.csv` | 41,930 | the day's demand per stop pair, before any bus could refuse it |
| `od_passengers.csv` | 11,173 | the passenger OD matrix, keyed exactly like `od_matrix.csv` |
| `od_passengers_timeline.csv` | 13,510 | the same matrix sliced by departure interval |
| `instants.csv` | 131,544 | the label-space ground truth: `AA`/`BB`/`BA`/`AB` counts per stop visit and 10-second instant |
| `od_passengers_wide_<operator>_<service>.csv` | — | the matrix as a dense origins × destinations grid, per operator and service day |
| `scenario.json` | — | every parameter, plus the sha256 of the supply tables it read |
| `report.json` | — | the counts, the invariants that were checked, and every warning |
| `stops.geojson`, `flows.geojson` | — | the day as GIS layers: points per stop, lines per OD relation |
| `flows_map.html` | — | the day on the ground: stops sized by boardings, OD lines by riders, with filters and a clock |
| `flows_cube.html` | — | the same day as a space-time cube: origin × destination × hour, entries and exits as two readings of one ride |
| `flows_replay.html` | — | one route's day played back: buses moving on the timetable, riders waiting as dots, boarding, alighting and who was left behind |

Totals, from that scenario:

```
trips          1429        stop visits    18792
stop pairs     41930       demand         17148
boarded        16419       alighted       16417
left behind    1252        never boarded  729
matrix cells   11173       timeline cells 13510
instants       131544      problems       0
```

24 % of the stop pairs carry anyone at all, the busiest single pair carries 53
riders — an inner-city Aveiro hop, the same shape the supply matrix shows — and
the mean load factor is 0.125, with 351 of the 18,792 stop visits at capacity.
A batch writes one row per scenario into `data/bus/scenarios/index.csv`, which is
the table to read when comparing scenarios:

```
uv run inmotion bus demand -n 3 --seed 42
```

| seed | policy | boardings/departure | demand | boarded | never boarded | problems |
|---|---|---|---|---|---|---|
| 42 | random | 20.6 | 29,473 | 26,862 | 2,611 | 0 |
| 43 | cycle | 23.1 | 33,079 | 29,342 | 3,737 | 0 |
| 44 | demand | 15.0 | 21,392 | 20,161 | 1,231 | 0 |

Two views are written per scenario, and they are two views of one dataset rather
than two reports. The map filters by service day, operator, bus line, minimum
riders, a stop search and an hour window; playing the day narrows the window and
the OD lines thin and thicken with the traffic under it. The cube lays the same
OD relation on a lattice — origin across, destination back, hours up — and the
point of it is the *two clocks*: a rider belongs to the hour they stepped on when
you read entries, and to the hour they stepped off when you read exits, and 312
of the 2,646 rides on the cube's lattice change hour between the two readings.
The lattice holds the busiest 60 of the 1,844 stops (`--cube-stops` moves that),
because every stop cubed is 1.7 million cells of mostly nothing; the page says
how many it left out.

A third page replays a single route. Pick a line and direction, and its buses move
along the line by their own timetables — interpolated between the two stops they
are between at the current clock — while each stop shows one dot per rider waiting
there, and a bus that calls adds `+N boarded`, `−N alighted` and, when it was full,
`N left` in red. The clock runs from the route's first departure to its last at 2,
8 or 30 minutes a second, and the scrubber jumps anywhere in the day.

Two kinds of trip cannot be replayed and are left out, counted in the page's note:
trips calling at a stop OpenStreetMap could not place (825 of the 1,429 trips are
affected by at least one of the two), and the handful whose scraped timetables run
backwards. A bus that disappears for two stops, or jumps back up the route, would
be a claim the data does not support. Of the 1,429 trips, 604 can be replayed, on
106 routes; a day at the default dial therefore looks sparse on any one route,
which is why a denser scenario (`--boardings-per-departure 40`) is the one to
watch.

## 2.5 Verification

Every scenario checks its own invariants before it is written, and a failure is
reported in `report.json`, printed, and reflected in the command's exit code
rather than left in a table that looks authoritative:

* `onboard_after = onboard_before + boarded` and `onboard_before = onboard_arrival − alighted`, at every stop of every trip;
* `onboard_after ≤ capacity` — no bus ever carries more riders than it has seats;
* `waiting_end = waiting_start + arrivals − boarded ≥ 0`, at every stop and hour — the queue conserves riders;
* Σ demand = Σ boarded + never boarded, over each operator and service day;
* the OD relation's total equals the flows' boardings, and the passenger matrix's total equals both;
* the intervals of the timeline sum back to the static matrix, cell by cell.

The pages themselves are checked by a browser, not by their source: the suite
renders both in headless Chrome and fails on an uncaught script error or on a
canvas with no colour on it. That check exists because the source-level checks
missed three real faults in one sitting — a helper the cube page never loaded, a
material that drew every cube black, and a camera fitted before the canvas had a
size — and a page that looks finished while drawing nothing is worse than one
that fails to build.

The supply itself is verified rather than asserted: every run checks that every
trip, every stop visit (with its clock), every trip's line and direction, and
every OD row (that its two ends are calls of that trip, in that order) trace back
to the scraped timetable, and `report.json` states what was checked. The demand is
the generated part, and the report says so in the same breath — the distinction
matters the moment a number from these tables is quoted somewhere else.

The one rule that exists twice — the per-trip stop pairs — is held to the
supply side's own implementation: `tests/test_bus_demand.py` asserts that the
polars enumeration equals `od.per_trip_pairs` row for row, on the toy network
and on all 158,184 rows of the scraped timetable. The same seed and the same
inputs produce byte-identical tables, which the suite also asserts.

## 2.6 Known limitations

* **Demand is synthetic, not calibrated.** Nothing here is a forecast of
  Aveiro's real ridership; the dials are conventional, and a scenario is a
  plausible day rather than a measured one.
* **A quarter of the stops have no coordinates.** They generate flows like any
  other stop — coordinates are only needed for the map — but the map shows
  1,389 of 1,844 stops and says so in the panel.
* **No transfers.** Demand is generated on the pairs the timetable serves
  *directly*, because that is the only relation the supply side can vouch for; a
  journey that needs a change is outside the model.
* **Buses fill exactly.** Boarding is capped at the free seats with no
  no-shows, so a saturated peak departure sits at a load factor of 1.00 rather
  than at 0.9-something.
* **One service day at a time.** Weekday and weekend are generated as separate
  days; a queue never boards a bus from the other service.
* **The cube is a reading aid, not the whole tensor.** It holds the busiest 60
  stops by boardings and the rides between them, which is 2.6 thousand points in
  the scenario above. Raising `--cube-stops` grows it quadratically; the page
  states what it omitted rather than pretending to be complete.
* **Which basemap the map draws, and why it is a choice.** OpenStreetMap's own
  tile servers answer a page opened from `file://` with 403 and their "Access
  blocked" page: their policy requires an HTTP `Referer`, which a local file
  never sends. CARTO's basemap CDN, tried next, appeared to demand an API key —
  and does, but not through the parameter their own placeholder names: it reads
  `?key=`, while `?api_key=` returns the same "API KEY REQUIRED" tile whether
  the key is valid or nonsense. With `?key=` it serves a real basemap, verified
  tile by tile before it was wired in. So the page takes `--basemap`: `auto`
  (CARTO when `CARTO_API_KEY` is set in `.env`, the timetable's own leg geometry
  otherwise — the default), `carto-dark`, `carto-light`, `network`, `osm` for a
  page served over HTTP, and `none`. The key is read from the environment and
  never written into `scenario.json`; it does end up inside the generated page,
  as a client-side basemap key must, which is why a scenario built with one
  should not be committed.
* Reading the CSVs needs `dtype` on the `line` column, exactly as the supply
  tables do: AveiroBus numbers its lines and also calls two of them `azul` and
  `verde`.

## 2.7 Reproducing

```bash
uv run inmotion bus geocode                      # coordinates, once
uv run inmotion bus demand -n 1 --seed 42        # the scenario above
uv run inmotion bus demand -n 3 --seed 42        # the batch table above
uv run pytest tests/test_bus_demand.py -q
```

Everything under `data/bus/raw/osm/` is a cache; the coordinate tables land in
`data/bus/processed/` next to the supply tables, and the scenarios in
`data/bus/scenarios/` (~40 MB each, regenerable from their `scenario.json`).

# 3. Quantization on the wavecom routers

The earlier quantization numbers were not wrong so much as *mixed*: the
registry and the HPO campaign load different files under the same family name.
`ts_jepa` resolves to `ts_jepa_ft_seed42.pt` (sha256 `c470d549…`, MCC 0.919)
while the old campaign loaded `full-after-hpo/ts_jepa_ft_seed42.pt`
(`45f9d3ed…`, MCC 0.491); `sigreg` is the reverse (`36b95242…`, 0.633 versus
`91e58334…`, 0.918). The two sets were never comparable. Section 3 uses the
registry checkpoints only and scores each exported artifact's own weights on
`dataset.csv` (703 test rows, split seed 42), with 2 000-resample paired
bootstrap CIs.

MCC (full table and CIs in `results/edge/tables/accuracy.csv`):

| model | fp32 | int8 | q8 | q6 | q5 | q4 | q3 | q2 | q1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lejepa | 0.914 | 0.912 | 0.914 | 0.912 | 0.908 | 0.908 | 0.838 | 0.323 | -0.001 |
| t_jepa | 0.898 | 0.898 | 0.898 | 0.896 | 0.896 | 0.886 | 0.786 | 0.339 | -0.020 |
| ts_jepa | 0.919 | 0.919 | 0.919 | 0.918 | 0.908 | 0.912 | 0.829 | 0.287 | -0.019 |
| cf_jepa | 0.908 | 0.906 | 0.906 | 0.906 | 0.905 | 0.905 | 0.899 | 0.453 | 0.134 |
| sigreg | 0.633 | 0.633 | 0.633 | 0.631 | 0.631 | 0.628 | 0.629 | 0.591 | 0.181 |

The nominal bit width is not what a router stores. Fixed per-group scales and
zero points collapse the nine schemes into five storage tiers: q2 costs the
same 6.3–6.8 bits/parameter as q4, and q1 the same 5.3–5.8 as q3, while scoring
far worse. **int8** (8.5–9.2 bits) is smaller than q8 (9.3–9.8) and loses
nothing; **q4** is the last 4-bit point whose paired ΔMCC CI contains zero on
all five models; **q3** costs the JEPA transformers 0.08–0.11 MCC but is free
for `cf_jepa` and `sigreg`; **q2 and q1 must not be shipped** (q1 is at chance
for four of five).

On-device: 45 artifacts × 2 routers = 90 replay cells, run on the actual
artifacts. Every cell passes its golden vectors, every predicted label equals
Python's (100 % agreement), |ΔMCC| = 0, and the largest logit deviation is
8.3e-06. The router computes the artifact, not an approximation of it.

Runtime cost, load time, RAM/CPU, core limiting, the labelled power model and
the router-interference study are in `results/edge/REPORT.md`.

```bash
./scripts/edge/run_campaign.sh --profile full --devices both
```
