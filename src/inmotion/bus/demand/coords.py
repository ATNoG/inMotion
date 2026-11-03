"""Stop coordinates, resolved from OpenStreetMap and never invented.

One cached Overpass request covers every bus stop of the region; matching then
happens locally, cheapest tier first. Each stop records the tier that placed it —
``manual``, ``exact``, ``expanded``, ``tokens``, ``fuzzy``, ``sibling``,
``nominatim`` — or ``none``, with the coordinates left blank and the stop listed
in ``stops_unresolved.csv``. A guessed position would put a phantom stop on the
map and a plausible one in an analysis, so a gap is reported, not filled.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from inmotion.bus import paths, sources
from inmotion.bus.od.stops import canonical

#: Overpass serves the index; Nominatim answers the per-stop fallback. Neither
#: is scraped, so both get the same user agent, retries and cache as the
#: operators, with a slower delay than the sites ask for.
OVERPASS = sources.Operator("osm", "OpenStreetMap (Overpass)", "https://overpass-api.de", 2.0)
NOMINATIM = sources.Operator(
    "osm-nominatim", "OpenStreetMap (Nominatim)", "https://nominatim.openstreetmap.org", 1.5
)
OVERPASS_URL = f"{OVERPASS.base}/api/interpreter"
NOMINATIM_URL = f"{NOMINATIM.base}/search"

#: The Aveiro district with a margin: both operators run inside it, and one
#: bounded request is what makes the matching local.
DEFAULT_BBOX = (40.30, -8.95, 41.10, -8.10)

#: Bus stops, bus stations and the platforms buses call at. The index is the
#: union: a platform is often named where the stop beside it is not.
INDEX_QUERY = """[out:json][timeout:{timeout}];
(
  nwr["highway"="bus_stop"]({bbox});
  nwr["amenity"="bus_station"]({bbox});
  nwr["public_transport"="platform"]["bus"="yes"]({bbox});
);
out center tags;"""

COORDS_FILE = "stop_coords.csv"
UNRESOLVED_FILE = "stops_unresolved.csv"
CLUSTERS_FILE = "stop_clusters.csv"

#: Abbreviations the timetables print and OSM spells out (``R.`` → ``rua``).
ABBREVIATIONS = {
    "r": "rua",
    "av": "avenida",
    "avn": "avenida",
    "lgo": "largo",
    "pq": "parque",
    "pct": "praceta",
    "est": "estacao",
    "dr": "doutor",
    "dra": "doutora",
    "eng": "engenheiro",
    "prof": "professor",
    "sr": "senhor",
    "sra": "senhora",
    "sta": "santa",
    "sto": "santo",
    "st": "santo",
    "n": "nossa",
    "qta": "quinta",
    "qt": "quinta",
    "pc": "praceta",
    "rot": "rotunda",
    "urb": "urbanizacao",
    "esc": "escola",
    "jd": "jardim",
    "ctr": "centro",
}

#: Trailing qualifiers for the two sides of one place: ``Barrocas A``/``B``.
SIDE_SUFFIXES = ("a", "b", "c", "1", "2", "3", "ida", "volta", "norte", "sul", "nascente", "poente")

#: How close two stops of different operators have to be to be one place.
CLUSTER_RADIUS_M = 60.0

EARTH_M_PER_DEGREE = 111_320.0


@dataclass(frozen=True)
class Resolution:
    """What the matcher did, per stop, and how the day's coordinates look."""

    frame: pl.DataFrame
    warnings: list[str]

    def counts(self) -> dict[str, int]:
        """How many stops came from each source, commonest first."""
        if not self.frame.height:
            return {}
        tallied = self.frame.group_by("source").len().sort("len", descending=True).iter_rows()
        return {str(row[0]): int(row[1]) for row in tallied}

    def summary(self) -> str:
        """One line per source, for the build report."""
        counts = self.counts()
        total = self.frame.height
        resolved = total - counts.get("none", 0)
        parts = [f"{name} {count:,}" for name, count in counts.items() if name != "none"]
        return f"{resolved:,}/{total:,} stops placed ({', '.join(parts) or 'nothing matched'})"


def expand(name: str) -> str:
    """The canonical form of a name, with its abbreviations spelled out."""
    words = []
    for word in canonical(name).split():
        words.append(ABBREVIATIONS.get(word, word))
    return " ".join(words)


def strip_parenthetical(name: str) -> str:
    """A name without its qualifier: ``Esmoriz (Avenida da Praia 3)``."""
    return re.sub(r"\(.*?\)", " ", name).strip()


def side_base(name: str) -> str:
    """A name without a trailing direction qualifier, for sibling matching."""
    expanded = expand(name)
    words = expanded.split()
    while len(words) > 1 and words[-1] in SIDE_SUFFIXES:
        words.pop()
    return " ".join(words)


def index_frame(payload: dict, *, kind: str = "node") -> pl.DataFrame:
    """The Overpass response as a table: one row per named stop."""
    rows = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        name = tags.get("name")
        lat = element.get("lat") or (element.get("center") or {}).get("lat")
        lon = element.get("lon") or (element.get("center") or {}).get("lon")
        if not name or lat is None or lon is None:
            continue
        rows.append(
            {
                "osm_id": f"{element.get('type', kind)}/{element.get('id')}",
                "name": str(name),
                "lat": float(lat),
                "lon": float(lon),
                "kind": str(
                    tags.get("highway") or tags.get("amenity") or tags.get("public_transport") or ""
                ),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "osm_id": pl.String,
            "name": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "kind": pl.String,
        },
        orient="row",
    )
    return frame.unique(subset=["osm_id"], keep="first") if frame.height else frame


def fetch_index(
    *,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    refresh: bool = False,
    timeout: int = 180,
    verbose: bool = True,
) -> pl.DataFrame:
    """Every named bus stop of the region, from the cache when it is there.

    This is the only network call the generator makes, and it happens in
    ``inmotion bus geocode`` rather than during a scenario run: scenarios are
    reproducible offline once the coordinates exist.
    """
    south, west, north, east = bbox
    query = INDEX_QUERY.format(bbox=f"{south},{west},{north},{east}", timeout=timeout)
    with sources.Fetcher(force=refresh, verbose=verbose) as fetcher:
        path = fetcher.post(
            OVERPASS_URL, {"data": query}, OVERPASS, directory=paths.osm_cache(), suffix=".json"
        )
    return index_frame(json.loads(path.read_text()))


def nominatim(
    name: str, *, bbox: tuple[float, float, float, float] = DEFAULT_BBOX, verbose: bool = False
) -> dict | None:
    """The best web match for one stop name, or nothing.

    Nominatim's usage policy is a request a second and an identifying agent, so
    this is only ever called for the stops the local index could not place, and
    each answer is cached like every other response.
    """
    south, west, north, east = bbox
    params = {
        "q": name,
        "format": "jsonv2",
        "limit": "1",
        "countrycodes": "pt",
        "viewbox": f"{west},{north},{east},{south}",
        "bounded": "1",
        "addressdetails": "0",
    }
    with sources.Fetcher(verbose=verbose) as fetcher:
        path = fetcher.post(
            NOMINATIM_URL,
            params,
            NOMINATIM,
            directory=paths.osm_cache(),
            suffix=".json",
        )
    try:
        results = json.loads(path.read_text())
    except json.JSONDecodeError:  # pragma: no cover - a cached error page
        return None
    for result in results if isinstance(results, list) else []:
        kind = str(result.get("type", "")) + " " + str(result.get("class", ""))
        if any(word in kind for word in ("bus", "platform", "station", "stop")):
            return {
                "osm_id": f"nominatim/{result.get('osm_id', '')}",
                "name": str(result.get("name") or name),
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "kind": str(result.get("type", "")),
            }
    return None


@dataclass(frozen=True)
class Index:
    """The OpenStreetMap stops, bucketed by every way a name can be compared."""

    rows: list[dict]
    exact: dict[str, list[int]]
    expanded: dict[str, list[int]]
    by_position: list[frozenset[str]]
    key_by_position: list[str]
    inverted: dict[str, list[int]]
    expanded_keys: list[str]

    def words_by_position(self, position: int) -> frozenset[str]:
        """The words of one OSM stop's name."""
        return self.by_position[position]

    def rare_candidates(self, query: set[str], *, limit: int = 5) -> list[int]:
        """Stops sharing a word that almost nothing else in the index uses.

        ``Arco do Comércio`` against OSM's ``Rua Comércio`` shares one rare word
        and nothing else. Anchoring on rarity is what keeps that from becoming
        ``Parque da Cidade`` against ``Café Cidade``: the word has to be
        distinctive before a loose string ratio is allowed to decide.
        """
        positions: dict[int, None] = {}
        for word in query:
            postings = self.inverted.get(word, ())
            if 0 < len(postings) <= limit:
                for position in postings:
                    positions.setdefault(position, None)
        return list(positions)

    def sharing_a_word(self, query: set[str]) -> list[int]:
        """Every stop that has at least one word in common with ``query``."""
        seen: dict[int, None] = {}
        for word in query:
            for position in self.inverted.get(word, ()):
                seen.setdefault(position, None)
        return list(seen)

    def containing(self, words: set[str], base: set[str]) -> set[int]:
        """Every stop whose name uses all of ``words``, or is used up by them.

        Forward: the timetable's words all appear in the OSM name. Reverse: the OSM name
        is contained in the timetable's, which is how OSM's ``Barrocas`` matches
        ``Barrocas A`` — allowed for one word only against the name without its direction
        suffix, so ``Aveiro`` cannot claim half the district.
        """
        if not words:
            return set()
        if all(word in self.inverted for word in words):
            forward = set.intersection(*(set(self.inverted[word]) for word in words))
        else:
            forward = set()
        reverse = {
            position
            for position in self.sharing_a_word(words)
            if self.by_position[position] <= words
            and (len(self.by_position[position]) >= 2 or self.by_position[position] <= base)
        }
        return forward | reverse


def build_index(index: pl.DataFrame) -> Index:
    """Bucket the OSM stops once, so each tier is a lookup rather than a scan."""
    rows = index.to_dicts() if index.height else []
    exact: dict[str, list[int]] = defaultdict(list)
    expanded: dict[str, list[int]] = defaultdict(list)
    by_position: list[frozenset[str]] = []
    key_by_position: list[str] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        exact[canonical(row["name"])].append(position)
        key = expand(row["name"])
        expanded[key].append(position)
        words = frozenset(key.split())
        by_position.append(words)
        key_by_position.append(key)
        for word in words:
            inverted[word].append(position)
    return Index(
        rows=rows,
        exact=exact,
        expanded=expanded,
        by_position=by_position,
        key_by_position=key_by_position,
        inverted=inverted,
        expanded_keys=sorted(expanded),
    )


def _nearest(
    candidates: Iterable[int],
    rows: list[dict],
    anchors: list[tuple[float, float]],
) -> int:
    """The candidate closest to where this operator's stops already are.

    Two OSM stops can share a name across a district; the one on the same line
    as the stops already placed is the one meant. With nothing placed yet the
    first candidate wins, which keeps the choice deterministic.
    """
    options = list(candidates)
    if len(options) == 1 or not anchors:
        return options[0]

    def distance(position: int) -> float:
        row = rows[position]
        return min((row["lat"] - lat) ** 2 + (row["lon"] - lon) ** 2 for lat, lon in anchors)

    return min(options, key=lambda position: (distance(position), position))


def match(
    stops: pl.DataFrame,
    index: pl.DataFrame,
    *,
    manual: pl.DataFrame | None = None,
    cutoff: float = 0.87,
    web: bool = False,
    verbose: bool = False,
) -> Resolution:
    """Place every stop of the network, or leave it blank and say so.

    The tiers run cheapest first and each one only sees what the previous could
    not place, so a stop that matches exactly never depends on the fuzzy
    cutoff — and every row carries the tier that placed it.
    """
    if not stops.height:
        return Resolution(
            frame=pl.DataFrame(schema=coords_schema()), warnings=["no stops to place"]
        )
    reference = build_index(index)
    rows = reference.rows
    placed: dict[str, dict] = {}
    warnings: list[str] = []

    if manual is not None and manual.height:
        for row in manual.iter_rows(named=True):
            placed[row["stop_id"]] = {
                "stop_id": row["stop_id"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "source": "manual",
                "osm_id": "",
                "score": 1.0,
            }

    order = stops.sort("operator", "stop").iter_rows(named=True)
    anchors: dict[str, list[tuple[float, float]]] = defaultdict(list)
    pending: list[dict] = []
    for stop in order:
        if stop["stop_id"] in placed:
            continue
        pending.append(stop)

    def place(stop: dict, row: dict, source: str, score: float) -> None:
        placed[stop["stop_id"]] = {
            "stop_id": stop["stop_id"],
            "lat": round(float(row["lat"]), 7),
            "lon": round(float(row["lon"]), 7),
            "source": source,
            "osm_id": row["osm_id"],
            "score": round(float(score), 3),
        }
        anchors[stop["operator"]].append((float(row["lat"]), float(row["lon"])))

    for pass_name in ("exact", "expanded", "tokens", "fuzzy"):
        still: list[dict] = []
        for stop in pending:
            name = stop["stop"]
            mine = anchors[stop["operator"]]
            if pass_name == "exact":
                key = canonical(name)
                if key in reference.exact:
                    place(stop, rows[_nearest(reference.exact[key], rows, mine)], "exact", 1.0)
                    continue
                base = canonical(strip_parenthetical(name))
                if base and base in reference.exact:
                    place(stop, rows[_nearest(reference.exact[base], rows, mine)], "exact", 0.95)
                    continue
            if pass_name == "expanded":
                key = expand(name)
                if key in reference.expanded:
                    place(
                        stop, rows[_nearest(reference.expanded[key], rows, mine)], "expanded", 0.9
                    )
                    continue
                base = expand(strip_parenthetical(name))
                if base and base in reference.expanded:
                    place(
                        stop, rows[_nearest(reference.expanded[base], rows, mine)], "expanded", 0.85
                    )
                    continue
            if pass_name == "tokens":
                # Both directions, because a timetable writes the direction of a
                # stop that OSM leaves out — "Barrocas A" against "Barrocas" —
                # and an OSM name that is contained in the timetable's is the
                # same place too. The reverse direction needs two words, or the
                # single word "Aveiro" would claim half the district.
                # Without the direction letter: "Loja do Cidadão A" is OSM's
                # "Forca - Loja do Cidadão", and the A is the timetable's way of
                # numbering the two sides of one place, not part of its name.
                base = set(expand(side_base(name)).split())
                hits = sorted(reference.containing(base, base)) if base else []
                if hits:
                    place(stop, rows[_nearest(hits, rows, mine)], "tokens", 0.8)
                    continue
            if pass_name == "fuzzy":
                key = expand(strip_parenthetical(name))
                # Only against names that share a word: a fuzzy match with no
                # shared word is a coincidence of letters, and comparing 4,000
                # keys per unplaced stop is a minute of work for those.
                candidates = sorted(
                    {
                        reference.key_by_position[position]
                        for position in reference.sharing_a_word(set(key.split()))
                    }
                )
                close = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
                if close:
                    place(
                        stop,
                        rows[_nearest(reference.expanded[close[0]], rows, mine)],
                        "fuzzy",
                        0.7,
                    )
                    continue
                anchored = sorted(
                    {
                        reference.key_by_position[position]
                        for position in reference.rare_candidates(set(key.split()))
                    }
                )
                loose = difflib.get_close_matches(key, anchored, n=1, cutoff=0.75)
                if loose:
                    place(
                        stop,
                        rows[_nearest(reference.expanded[loose[0]], rows, mine)],
                        "fuzzy",
                        0.6,
                    )
                    continue
            still.append(stop)
        pending = still

    # One side of a place fixes the other: "Barrocas A" and "Barrocas B" are the
    # same corner, and the timetable names both.
    resolved_bases = {
        (key["operator"], side_base(key["stop"])): key
        for key in (dict(row) for row in stops.iter_rows(named=True))
        if key["stop_id"] in placed
    }
    still = []
    for stop in pending:
        sibling = resolved_bases.get((stop["operator"], side_base(stop["stop"])))
        if sibling is not None and sibling["stop_id"] != stop["stop_id"]:
            placed[stop["stop_id"]] = {
                "stop_id": stop["stop_id"],
                "lat": placed[sibling["stop_id"]]["lat"],
                "lon": placed[sibling["stop_id"]]["lon"],
                "source": "sibling",
                "osm_id": placed[sibling["stop_id"]]["osm_id"],
                "score": 0.6,
            }
        else:
            still.append(stop)
    pending = still

    if web:
        still = []
        for stop in pending:
            found = nominatim(f"{stop['stop']}, Aveiro, Portugal", verbose=verbose)
            if found is None:
                still.append(stop)
                continue
            place(stop, found, "nominatim", 0.5)
        pending = still

    for stop in pending:
        placed[stop["stop_id"]] = {
            "stop_id": stop["stop_id"],
            "lat": None,
            "lon": None,
            "source": "none",
            "osm_id": "",
            "score": 0.0,
        }

    frame = pl.DataFrame(
        [placed[stop["stop_id"]] for stop in stops.iter_rows(named=True)],
        schema=coords_schema(),
        orient="row",
    )
    unnamed = int(index.height == 0)
    if unnamed:
        warnings.append("the OpenStreetMap index is empty: nothing could be matched")
    missing = int(frame.filter(pl.col("source") == "none").height)
    if missing:
        warnings.append(
            f"{missing:,} stops have no coordinates; they are listed in {UNRESOLVED_FILE}"
        )
    return Resolution(frame=frame, warnings=warnings)


def coords_schema() -> dict:
    """The coordinate table's columns and types."""
    return {
        "stop_id": pl.String,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "source": pl.String,
        "osm_id": pl.String,
        "score": pl.Float64,
    }


def cluster(frame: pl.DataFrame, *, radius_m: float = CLUSTER_RADIUS_M) -> pl.DataFrame:
    """Group stops that are the same place, whichever operator runs them.

    The two networks share no stop id, so the same corner appears twice in the
    tables and would appear twice on the map. A grid over the placed stops is
    enough: 1,800 points do not need a spatial index, and the union of the
    cells within the radius is exactly the connected components wanted here.
    """
    if not frame.height:
        return frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("cluster_id"))
    rows = frame.to_dicts()
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    scale = radius_m / EARTH_M_PER_DEGREE
    for position, row in enumerate(rows):
        if row["lat"] is None or row["lon"] is None:
            continue
        lat = float(row["lat"])
        lon = float(row["lon"]) * math.cos(math.radians(lat))
        cells[(int(lat // scale), int(lon // scale))].append(position)
    for (cell_lat, cell_lon), members in cells.items():
        for delta_lat in (-1, 0, 1):
            for delta_lon in (-1, 0, 1):
                neighbours = cells.get((cell_lat + delta_lat, cell_lon + delta_lon), [])
                for left in members:
                    for right in neighbours:
                        if left >= right:
                            continue
                        a, b = rows[left], rows[right]
                        if _metres(a, b) <= radius_m:
                            union(left, right)
    components: dict[int, int] = {}
    ids = []
    for position in range(len(rows)):
        root = find(position)
        ids.append(components.setdefault(root, len(components)))
    return frame.with_columns(pl.Series("cluster_id", ids, dtype=pl.Int64))


def _metres(a: dict, b: dict) -> float:
    """Flat-earth distance between two rows, which is all a 60 m radius needs."""
    mean_lat = math.radians((float(a["lat"]) + float(b["lat"])) / 2)
    dx = (float(a["lon"]) - float(b["lon"])) * EARTH_M_PER_DEGREE * math.cos(mean_lat)
    dy = (float(a["lat"]) - float(b["lat"])) * EARTH_M_PER_DEGREE
    return math.hypot(dx, dy)


def write(frame: pl.DataFrame, directory: str | Path = paths.PROCESSED_DIR) -> list[Path]:
    """Write the coordinate table, the clusters and the unresolved list."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    clustered = cluster(frame)
    coords = root / COORDS_FILE
    clustered.write_csv(coords)
    written.append(coords)
    clusters = (
        clustered.filter(pl.col("cluster_id").is_not_null())
        .group_by("cluster_id")
        .agg(
            pl.len().alias("stops"),
            pl.col("lat").mean().alias("lat"),
            pl.col("lon").mean().alias("lon"),
            pl.col("operator").unique().sort().str.join(" ").alias("operators"),
        )
        .sort("stops", descending=True)
    )
    cluster_path = root / CLUSTERS_FILE
    clusters.write_csv(cluster_path)
    written.append(cluster_path)
    unresolved = clustered.filter(pl.col("source") == "none")
    unresolved_path = root / UNRESOLVED_FILE
    unresolved.write_csv(unresolved_path)
    written.append(unresolved_path)
    return written


def geocode(
    *,
    source_dir: str | Path = paths.PROCESSED_DIR,
    attributes: str | Path | None = None,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    refresh: bool = False,
    web: bool = False,
    verbose: bool = True,
) -> Resolution:
    """Place every stop of the network, writing the tables `bus demand` reads."""
    from inmotion.bus.demand import inputs

    root = Path(source_dir)
    network = inputs.read_network(root)
    stops = network.dictionary().select("stop_id", "operator", "stop")
    given = inputs.read_stop_attributes(attributes)
    manual = (
        given.select("stop_id", "lat", "lon").filter(
            pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
        )
        if given.height and "lat" in given.columns
        else None
    )
    index = fetch_index(bbox=bbox, refresh=refresh, verbose=verbose)
    resolution = match(stops, index, manual=manual, web=web, verbose=verbose)
    enriched = resolution.frame.join(stops.select("stop_id", "operator"), on="stop_id", how="left")
    write(enriched, root)
    return Resolution(frame=enriched, warnings=resolution.warnings)
