"""The generated day as GIS layers, and as the payloads the pages draw.

Coordinates come from ``coords``, so only placed stops appear. Two GeoJSON files
serve a GIS; the rest feeds the three viewers in ``pages``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from inmotion.bus.demand import config as scenario_config
from inmotion.bus.demand import pages
from inmotion.bus.demand.config import ScenarioConfig
from inmotion.bus.od.bands import format_minutes

#: What the page draws underneath the flows. The keys are the values
#: :data:`inmotion.bus.demand.config.BASEMAPS` allows, and a test holds the two
#: tables together.
BASE_LAYERS: dict[str, dict[str, str]] = {
    # CARTO's basemaps, which require an API key. The query parameter really is
    # `key`: with `api_key` their CDN answers 200 with a tile that reads "API KEY
    # REQUIRED", and a wrong key gives the same tile — which is why the wrong
    # parameter looks like a broken key rather than a broken URL.
    "carto-dark": {
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key={key}",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
        "subdomains": "abcd",
        "max_zoom": "20",
        "key_env": scenario_config.CARTO_KEY_ENV,
    },
    "carto-light": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?key={key}",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
        "subdomains": "abcd",
        "max_zoom": "20",
        "key_env": scenario_config.CARTO_KEY_ENV,
    },
    # The network's own legs, drawn from the timetable: no tiles, no key, and it
    # works from a local file.
    "network": {
        "url": "",
        "attribution": "the scraped timetable",
        "subdomains": "",
        "max_zoom": "19",
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
        "subdomains": "",
        "max_zoom": "19",
        "note": "OpenStreetMap's tiles are refused to a page opened from a local file; "
        "serve this directory over HTTP (python3 -m http.server) to see them.",
    },
    "none": {"url": "", "attribution": "none", "subdomains": "", "max_zoom": "19"},
}

#: How many OD lines the page draws. The whole relation is tens of thousands of
#: pairs, most of them carrying one rider; the map is for the ones that matter.
DEFAULT_FLOW_LIMIT = 400

STOPS_FILE = "stops.geojson"
FLOWS_FILE = "flows.geojson"
PAGE_FILE = "flows_map.html"
CUBE_FILE = "flows_cube.html"
REPLAY_FILE = "flows_replay.html"


def stop_features(stops: pl.DataFrame, coords: pl.DataFrame, flows: pl.DataFrame) -> list[dict]:
    """One point per placed stop, carrying what the flows say about it."""
    placed = coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).select(
        "stop_id", "lat", "lon", "source", "cluster_id", pl.col("operator").alias("coord_operator")
    )
    boarded = (
        flows.group_by("stop_id").agg(
            pl.col("boarded").sum().alias("boarded"),
            pl.col("alighted").sum().alias("alighted"),
            pl.col("denied").sum().alias("denied"),
            pl.col("waiting_at_arrival").max().alias("peak_queue"),
        )
        if flows.height
        else pl.DataFrame(
            schema={
                "stop_id": pl.String,
                "boarded": pl.Int64,
                "alighted": pl.Int64,
                "denied": pl.Int64,
                "peak_queue": pl.Int64,
            }
        )
    )
    joined = (
        stops.select("stop_id", "operator", "stop", "lines")
        .join(placed, on="stop_id", how="inner")
        .join(boarded, on="stop_id", how="left")
        .with_columns(
            pl.col("boarded").fill_null(0),
            pl.col("alighted").fill_null(0),
            pl.col("denied").fill_null(0),
            pl.col("peak_queue").fill_null(0),
            pl.col("cluster_id").fill_null(-1),
        )
        .sort("boarded", descending=True)
    )
    return [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "stop_id": row["stop_id"],
                "name": row["stop"],
                "operator": row["operator"],
                "lines": row["lines"],
                "boarded": int(row["boarded"]),
                "alighted": int(row["alighted"]),
                "denied": int(row["denied"]),
                "peak_queue": int(row["peak_queue"]),
                "cluster_id": int(row["cluster_id"]),
                "source": row["source"],
            },
        }
        for row in joined.iter_rows(named=True)
    ]


def flow_features(
    od: pl.DataFrame, coords: pl.DataFrame, *, limit: int = DEFAULT_FLOW_LIMIT
) -> list[dict]:
    """The busiest OD relations as straight lines between their two stops.

    A straight line is not the route a bus takes and does not pretend to be one:
    it is the pair a rider travelled, drawn between the two stops it names, which
    is what an OD relation is.
    """
    if not od.height:
        return []
    placed = coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).select(
        "stop_id", "lat", "lon"
    )
    relation = (
        od.group_by("origin_id", "origin", "destination_id", "destination", "service_group")
        .agg(
            pl.col("passengers").sum(),
            # The lines that carried these riders, for the viewer's filter. Capped:
            # a cell served by thirty lines does not need all thirty names, and the
            # payload is read by a browser.
            pl.col("line").unique().sort().head(12).str.join(" ").alias("lines"),
            pl.col("operator").first().alias("operator"),
            # The hourly shape, so playing the day on the map is a filter over
            # this rather than a second pass through the flow tables.
            pl.struct(
                hour=pl.col("departure_minutes") // 60,
                riders=pl.col("passengers"),
            ).alias("slots"),
        )
        .sort("passengers", descending=True)
        .head(limit)
    )
    joined = (
        relation.join(
            placed.rename({"stop_id": "origin_id", "lat": "origin_lat", "lon": "origin_lon"}),
            on="origin_id",
            how="inner",
        )
        .join(
            placed.rename(
                {"stop_id": "destination_id", "lat": "destination_lat", "lon": "destination_lon"}
            ),
            on="destination_id",
            how="inner",
        )
        .sort("passengers", descending=True)
    )
    return [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [row["origin_lon"], row["origin_lat"]],
                    [row["destination_lon"], row["destination_lat"]],
                ],
            },
            "properties": {
                "from": row["origin"],
                "to": row["destination"],
                "origin_id": row["origin_id"],
                "destination_id": row["destination_id"],
                "service_group": row["service_group"],
                "operator": row["operator"],
                "lines": row["lines"],
                "hours": _hourly(row["slots"]),
                "passengers": int(row["passengers"]),
            },
        }
        for row in joined.iter_rows(named=True)
    ]


def network_legs(flows: pl.DataFrame, coords: pl.DataFrame) -> list[list[list[float]]]:
    """Every leg between two consecutive stops of a trip, once, with coordinates.

    This is what the page draws when no tiles are wanted: the shape of the
    network as the timetable describes it. It is not a street map and does not
    pretend to be one — it is the geometry the flows actually live on, and it
    costs nothing but the data already loaded.
    """
    if not flows.height:
        return []
    placed = coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).select(
        "stop_id", "lat", "lon"
    )
    if not placed.height:
        return []
    ordered = flows.sort("trip_id", "stop_seq").select("trip_id", "stop_id")
    legs = (
        ordered.with_columns(pl.col("stop_id").shift(-1).over("trip_id").alias("next_stop"))
        .drop_nulls("next_stop")
        .filter(pl.col("stop_id") != pl.col("next_stop"))
        .unique(subset=["stop_id", "next_stop"])
    )
    joined = legs.join(
        placed.rename({"stop_id": "stop_id", "lat": "lat_a", "lon": "lon_a"}), on="stop_id"
    ).join(placed.rename({"stop_id": "next_stop", "lat": "lat_b", "lon": "lon_b"}), on="next_stop")
    return [
        [[row["lon_a"], row["lat_a"]], [row["lon_b"], row["lat_b"]]]
        for row in joined.iter_rows(named=True)
    ]


def hour_window(od: pl.DataFrame) -> dict:
    """The hours the day touches, and their labels, past midnight included.

    A day that starts at 04:30 and ends at 01:20 needs both ends on one axis, and
    the labels keep the operators' own convention of printing the small hours as
    25:00 rather than wrapping to 01:00.
    """
    if not od.height:
        return {"min": 0, "max": 0, "labels": ["00:00"]}
    low = int(od["departure_minutes"].min()) // 60
    high = int(od["arrival_minutes"].max()) // 60
    return {
        "min": low,
        "max": high,
        "labels": [format_minutes(hour * 60) for hour in range(high + 1)],
    }


def _hourly(slots: list[dict]) -> list[list[int]]:
    """The slots of one relation as ``[[hour, riders], ...]``, summed per hour."""
    tally: dict[int, int] = {}
    for slot in slots:
        hour = int(slot["hour"])
        tally[hour] = tally.get(hour, 0) + int(slot["riders"])
    return [[hour, tally[hour]] for hour in sorted(tally)]


def basemap_payload(environment: dict[str, str] | None = None) -> list[dict]:
    """Every basemap the viewer may switch to, and whether this machine can draw it.

    The key is injected here and nowhere else: the page carries it, the scenario
    record does not.
    """
    variables = os.environ if environment is None else environment
    payload = []
    for name, layer in BASE_LAYERS.items():
        entry = {
            "name": name,
            "url": layer["url"],
            "attribution": layer["attribution"],
            "subdomains": layer["subdomains"],
            "max_zoom": int(layer["max_zoom"]),
            "available": True,
            "note": layer.get("note", ""),
        }
        variable = layer.get("key_env")
        if variable:
            key = variables.get(variable)
            entry["available"] = bool(key)
            if key:
                entry["url"] = entry["url"].replace("{key}", key)
            else:
                entry["note"] = f"needs {variable} in .env"
        payload.append(entry)
    return payload


def placed_count(coords: pl.DataFrame) -> int:
    """How many stops have a position, which is what the map needs to exist."""
    if not coords.height or "lat" not in coords.columns:
        return 0
    return coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).height


def require_key(basemap: str, environment: dict[str, str] | None = None) -> None:
    """Refuse a keyed basemap whose key this machine does not have.

    Called before anything is written, so a run that cannot draw its map says so
    instead of leaving a page full of "API KEY REQUIRED" tiles behind.
    """
    layer = BASE_LAYERS.get(basemap, {})
    variable = layer.get("key_env")
    if not variable:
        return
    variables = os.environ if environment is None else environment
    if not variables.get(variable):
        raise RuntimeError(
            f"the {basemap} basemap needs an API key in ${variable}. Put it in .env "
            f"(gitignored) and `set -a && . ./.env && set +a`, or use --basemap network "
            f"for the timetable's own geometry, or --basemap none for no basemap."
        )


def cube_payload(
    od: pl.DataFrame,
    flows: pl.DataFrame,
    stops: pl.DataFrame,
    coords: pl.DataFrame,
    *,
    stops_limit: int = 60,
) -> dict:
    """The OD relation as a space-time cube: origin, destination, and the clock.

    Each point is one ride, placed at its origin and destination with *two* clocks:
    when they boarded and when they got off. That is the point of the view — entries
    belong to the hour a rider stepped on, exits to the hour they stepped off, and a
    20-minute ride crosses both.

    The lattice holds the busiest ``stops_limit`` stops with coordinates: every stop
    cubed is 1.7 million cells of mostly nothing. What is left out is stated, so the
    page can say it.
    """
    if not od.height or not flows.height:
        return {"stops": [], "points": [], "lines": [], "operators": [], "groups": []}
    activity = flows.group_by("stop_id").agg(
        pl.col("boarded").sum().alias("boarded"),
        pl.col("alighted").sum().alias("alighted"),
        pl.col("waiting_at_arrival").max().alias("peak_queue"),
    )
    placed = (
        stops.select("stop_id", "stop", "operator", "lines")
        .join(
            coords.filter(pl.col("lat").is_not_null()).select("stop_id", "lat", "lon"), on="stop_id"
        )
        .join(activity, on="stop_id", how="left")
        .with_columns(pl.col("boarded").fill_null(0), pl.col("alighted").fill_null(0))
        .sort(["boarded", "alighted"], descending=[True, True])
        .head(stops_limit)
        .with_columns(pl.int_range(pl.len()).alias("index"))
    )
    if not placed.height:
        return {"stops": [], "points": [], "lines": [], "operators": [], "groups": []}
    index_of = dict(placed.select("stop_id", "index").iter_rows())
    lines = sorted(od["line"].unique().to_list())
    line_index = {line: position for position, line in enumerate(lines)}
    operators = sorted(od["operator"].unique().to_list())
    groups = sorted(od["service_group"].unique().to_list())
    points = []
    for row in od.iter_rows(named=True):
        origin = index_of.get(row["origin_id"])
        destination = index_of.get(row["destination_id"])
        if origin is None or destination is None:
            continue
        points.append(
            {
                "o": origin,
                "d": destination,
                "b": int(row["departure_minutes"]) // 60,
                "a": int(row["arrival_minutes"]) // 60,
                "n": int(row["passengers"]),
                "l": line_index[row["line"]],
                "p": operators.index(row["operator"]),
                "s": groups.index(row["service_group"]),
            }
        )
    return {
        "stops": [
            {
                "id": row["stop_id"],
                "name": row["stop"],
                "operator": row["operator"],
                "lines": row["lines"],
                "boarded": int(row["boarded"]),
                "alighted": int(row["alighted"]),
                "peak_queue": int(row["peak_queue"] or 0),
                "lat": row["lat"],
                "lon": row["lon"],
            }
            for row in placed.iter_rows(named=True)
        ],
        "points": points,
        "lines": lines,
        "operators": operators,
        "groups": groups,
        "omitted": max(stops.height - placed.height, 0),
    }


def replay_payload(flows: pl.DataFrame, stops: pl.DataFrame, coords: pl.DataFrame) -> dict:
    """One bus route's day, as an animation the page can drive.

    Only trips whose every stop has a position are replayed: a bus that vanishes for
    two stops and reappears elsewhere is worse than no bus. The page says how many
    trips that leaves.
    """
    placed = coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()).select(
        "stop_id", "lat", "lon"
    )
    if not flows.height or not placed.height:
        return {"routes": [], "stops": [], "omitted": 0}
    have = (
        flows.select("trip_id")
        .join(
            flows.with_columns(pl.col("stop_id").is_in(placed["stop_id"]).alias("placed"))
            .group_by("trip_id")
            .agg(pl.col("placed").all().alias("complete")),
            on="trip_id",
        )
        .unique()
    )
    complete = set(have.filter(pl.col("complete"))["trip_id"].to_list())
    usable = flows.filter(pl.col("trip_id").is_in(complete)).sort("trip_id", "stop_seq")
    if not usable.height:
        return {"routes": [], "stops": [], "omitted": flows["trip_id"].n_unique()}

    used_stops = usable["stop_id"].unique().to_list()
    index = {stop_id: position for position, stop_id in enumerate(sorted(used_stops))}
    positions = (
        stops.select("stop_id", "stop")
        .filter(pl.col("stop_id").is_in(used_stops))
        .join(placed, on="stop_id")
        .sort("stop_id")
    )
    stop_table = [
        {"id": row["stop_id"], "name": row["stop"], "lat": row["lat"], "lon": row["lon"]}
        for row in positions.iter_rows(named=True)
    ]

    grouped: dict[str, dict] = {}
    for row in usable.iter_rows(named=True):
        key = f"{row['operator']}|{row['line']}|{row['direction']}|{row['service_group']}"
        route = grouped.setdefault(
            key,
            {
                "key": key,
                "operator": row["operator"],
                "line": row["line"],
                "direction": row["direction"],
                "service": row["service_group"],
                "trips": [],
                "trips_all": 0,
            },
        )
        events = route["trips"] if False else None
        del events
        trip = route["trips"]
        if not trip or trip[-1]["id"] != row["trip_id"]:
            trip.append(
                {
                    "id": row["trip_id"],
                    "capacity": int(row["capacity"]),
                    "bus": row["bus_size"],
                    "events": [],
                }
            )
        trip[-1]["events"].append(
            [
                index[row["stop_id"]],
                int(row["minutes"]),
                int(row["boarded"]),
                int(row["alighted"]),
                int(row["onboard_after"]),
                int(row["waiting_at_arrival"]),
                int(row["denied"]),
            ]
        )

    backwards = 0
    for route in grouped.values():
        kept = []
        for trip in route["trips"]:
            minutes = [event[1] for event in trip["events"]]
            if minutes == sorted(minutes):
                kept.append(trip)
            else:
                # The scraped timetables contain a few trips whose times run
                # backwards — the operator prints the last trip of the day
                # without rolling it past midnight. A replayed bus would jump
                # back down the route, so the trip is left out, not "fixed".
                backwards += 1
        route["trips"] = kept
    grouped = {key: route for key, route in grouped.items() if route["trips"]}
    for route in grouped.values():
        route["stops"] = sorted({event[0] for trip in route["trips"] for event in trip["events"]})
        # The full pattern: the longest trip's own stop order, which is what the
        # line is drawn along. Taking the union of every trip's stops would draw
        # a route that no single bus runs.
        route["shape"] = max((trip["events"] for trip in route["trips"]), key=len)
        route["shape"] = [event[0] for event in route["shape"]]
        route["boarded"] = sum(event[2] for trip in route["trips"] for event in trip["events"])
        route["label"] = f"{route['line']} · {route['direction']} · {route['service']}"
        route["trips_all"] = int(
            flows.filter(
                (pl.col("operator") == route["operator"])
                & (pl.col("line") == route["line"])
                & (pl.col("direction") == route["direction"])
                & (pl.col("service_group") == route["service"])
            )["trip_id"].n_unique()
        )
        route["day"] = {
            "from": min(trip["events"][0][1] for trip in route["trips"]),
            "to": max(trip["events"][-1][1] for trip in route["trips"]),
        }
        route["trips"].sort(key=lambda trip: trip["events"][0][1])
    ordered = sorted(grouped.values(), key=lambda route: (-route["boarded"], route["key"]))
    return {
        "routes": ordered,
        "stops": stop_table,
        "omitted": int(flows["trip_id"].n_unique() - len(complete) + backwards),
        "backwards": backwards,
    }


def write_replay(
    directory: Path,
    *,
    flows: pl.DataFrame,
    stops: pl.DataFrame,
    coords: pl.DataFrame,
    config: ScenarioConfig,
    report,
) -> list[Path]:
    """Write the route-replay page, when there is a route worth replaying."""
    payload = replay_payload(flows, stops, coords)
    if not payload["routes"]:
        return []
    basemaps = basemap_payload()
    chosen = next(entry for entry in basemaps if entry["name"] == config.basemap)
    page = directory / REPLAY_FILE
    page.write_text(
        pages.render_replay(payload, basemaps=basemaps, chosen=chosen, config=config, report=report)
    )
    return [page]


def write_geojson(path: Path, features: list[dict]) -> Path:
    """A FeatureCollection, written the way any GIS expects to read it."""
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False) + "\n"
    )
    return path


def write(
    directory: Path,
    *,
    flows: pl.DataFrame,
    od: pl.DataFrame,
    stops: pl.DataFrame,
    coords: pl.DataFrame,
    matrix: pl.DataFrame,
    config: ScenarioConfig,
    report,
    limit: int = DEFAULT_FLOW_LIMIT,
) -> list[Path]:
    """Write the two GeoJSON files and the page, when coordinates allow it.

    Two tables, because they answer different questions: the stop points are
    sized by what the *trip flows* say happened at each stop, and the lines are
    drawn from the *OD relation*, which is the only table that knows where a
    rider who boarded got off.
    """
    placed = coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    if not placed.height:
        return []
    stop_points = stop_features(stops, coords, flows)
    lines = flow_features(od, coords, limit=limit)
    legs = network_legs(flows, coords)
    written = [
        write_geojson(directory / STOPS_FILE, stop_points),
        write_geojson(directory / FLOWS_FILE, lines),
    ]
    basemaps = basemap_payload()
    chosen = next(entry for entry in basemaps if entry["name"] == config.basemap)
    page = directory / PAGE_FILE
    page.write_text(
        pages.render_map(
            stop_points,
            lines,
            legs,
            basemaps=basemaps,
            chosen=chosen,
            hours=hour_window(od),
            lines=sorted(od["line"].unique().to_list()),
            operators=sorted(od["operator"].unique().to_list()),
            groups=sorted(od["service_group"].unique().to_list()),
            config=config,
            report=report,
            matrix=matrix,
        )
    )
    written.append(page)
    return written


def write_cube(
    directory: Path,
    *,
    od: pl.DataFrame,
    flows: pl.DataFrame,
    stops: pl.DataFrame,
    coords: pl.DataFrame,
    config: ScenarioConfig,
    report,
    stop_total: int,
) -> Path:
    """Write the space-time cube page, when coordinates allow one."""
    cube = cube_payload(od, flows, stops, coords, stops_limit=config.cube_stops)
    if not cube["points"]:
        return directory / CUBE_FILE
    hours = [point["b"] for point in cube["points"]] + [point["a"] for point in cube["points"]]
    window = {
        "min": min(hours),
        "max": max(hours),
        "labels": [f"{hour % 24:02d}:00" for hour in range(max(hours) + 1)],
    }
    shifted = sum(1 for point in cube["points"] if point["b"] != point["a"])
    page = directory / CUBE_FILE
    page.write_text(
        pages.render_cube(
            cube,
            config,
            report,
            hours=window,
            shifted=shifted,
            stop_total=stop_total,
        )
    )
    return page
