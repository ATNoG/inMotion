"""The passenger-flow generator: the demand model, the queue, and the matrices.

The supply side of the network is already tested in ``tests/test_bus.py``; what
is tested here is what the generator does with it. The one rule that exists
twice — the per-trip stop pairs — is held to ``od.per_trip_pairs`` row for row,
on the toy network and on the scraped table when it is present, so a second
implementation cannot quietly become a second answer.

Nothing here touches the network: the OpenStreetMap responses are fixtures, and
the geocoding tests are skipped when the cached index is absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from PIL import Image

from inmotion.bus import od, paths
from inmotion.bus.demand import (
    allocate,
    assignment,
    build,
    config,
    coords,
    inputs,
    matrices,
    model,
    pairs,
    validate,
)
from inmotion.bus.demand import (
    map as map_module,
)
from inmotion.bus.demand import pages as pages_module
from inmotion.bus.demand.instants import instants as instant_rows
from inmotion.bus.demand.simulate import Flows as FlowDay
from inmotion.bus.demand.simulate import simulate as run_day

TOY_COLUMNS = {
    "trip_id": pl.String,
    "operator": pl.String,
    "line": pl.String,
    "direction": pl.String,
    "service": pl.String,
    "service_group": pl.String,
    "stop_seq": pl.Int64,
    "stop_id": pl.String,
    "stop": pl.String,
    "minutes": pl.Int64,
    "served": pl.Boolean,
}


def toy_stop_times(rows: list[tuple]) -> pl.DataFrame:
    """A tiny timetable, from ``(trip, seq, stop, minutes)`` rows.

    ``stop`` is the bare name; the id is derived the way the build derives it,
    so the toy tables key on the same stop ids the real ones do.
    """
    return pl.DataFrame(
        [
            {
                "trip_id": trip,
                "operator": "toy",
                "line": "1",
                "direction": "ida",
                "service": "weekday",
                "service_group": "weekday",
                "stop_seq": seq,
                "stop_id": f"toy:{name.lower()}",
                "stop": name,
                "minutes": minutes,
                "served": minutes is not None,
            }
            for trip, seq, name, minutes in rows
        ],
        schema=TOY_COLUMNS,
    )


@pytest.fixture
def toy_timetable() -> pl.DataFrame:
    """Four stops' worth of edge cases.

    ``t1`` is a plain out-and-back, ``t2`` serves the same stops in another
    order, ``t3`` names one stop twice in the same minute, ``t4`` skips a stop,
    and ``t5`` starts in the morning peak and finishes in the midday — the trip
    whose first stop's band is not its pairs' band.
    """
    return toy_stop_times(
        [
            ("t1", 1, "Alpha", 480),
            ("t1", 2, "Beta", 485),
            ("t1", 3, "Gamma", 500),
            ("t2", 1, "Alpha", 600),
            ("t2", 2, "Gamma", 620),
            ("t2", 3, "Beta", 630),
            ("t3", 1, "Alpha", 700),
            ("t3", 2, "Alpha", 700),
            ("t3", 3, "Beta", 710),
            ("t4", 1, "Beta", 720),
            ("t4", 2, "Gamma", None),
            ("t4", 3, "Alpha", 740),
            ("t5", 1, "Beta", 560),
            ("t5", 2, "Gamma", 575),
            ("t5", 3, "Alpha", 590),
        ]
    )


# -- configuration ---------------------------------------------------------


def test_a_bus_size_parses_from_the_cli_notation() -> None:
    sizes = config.BusSize.parse("standard:40:8, minibus:20:2 ,articulated:70:3")
    assert [size.name for size in sizes] == ["minibus", "standard", "articulated"]
    assert [size.capacity for size in sizes] == [20, 40, 70]
    assert (sizes[1].name, sizes[1].capacity, sizes[1].count) == ("standard", 40, 8)


@pytest.mark.parametrize("text", ["standard", "standard:40", "standard:many:2", "", " , "])
def test_a_fleet_that_cannot_be_read_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="bus size|needs at least one"):
        config.BusSize.parse(text)


def test_a_fleet_with_no_buses_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no buses"):
        config.ScenarioConfig(fleet=(config.BusSize("standard", 40, 0),))


def test_a_scenario_round_trips_through_its_own_json() -> None:
    original = config.ScenarioConfig(
        seed=7,
        service_groups=("weekday", "weekend"),
        fleet=config.DEFAULT_FLEET,
        fleet_policy="demand",
        daily_boardings=1200,
        alpha=0.9,
        beta=0.05,
    )
    payload = json.loads(json.dumps(original.to_dict()))
    restored = config.ScenarioConfig.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.resolved_label() == "s0007-demand-daily1200-weekday-weekend"
    assert restored.fingerprint() == original.fingerprint()


def test_a_different_parameter_is_a_different_fingerprint() -> None:
    base = config.ScenarioConfig()
    assert base.fingerprint() != config.ScenarioConfig(beta=0.07).fingerprint()


def test_a_sample_has_one_seed_and_one_label_per_scenario() -> None:
    space = config.ScenarioSpace(scenarios=5, seed=11)
    drawn = space.sample()
    assert len(drawn) == 5
    assert len({scenario.resolved_label() for scenario in drawn}) == 5
    assert len({scenario.seed for scenario in drawn}) == 5
    assert [scenario.seed for scenario in drawn] == [11, 12, 13, 14, 15]
    mixes = {
        tuple((size.name, size.capacity, size.count) for size in scenario.fleet)
        for scenario in drawn
    }
    assert len(mixes) > 1
    assert all(sum(s.band_weights.values()) == pytest.approx(1.0) for s in drawn)


def test_a_peak_share_moves_the_peaks_and_keeps_the_day_whole() -> None:
    scaled = config.with_peak_share(config.DEFAULT_BAND_WEIGHTS, 0.8)
    assert sum(scaled.values()) == pytest.approx(1.0)
    assert scaled["am_peak"] + scaled["pm_peak"] == pytest.approx(0.8)
    assert scaled["midday"] < scaled["am_peak"]


def test_a_sweep_crosses_its_values_into_a_grid(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"beta": [0.02, 0.1], "fleet_policy": ["random", "cycle"]}))
    drawn = config.sweep_configs(sweep, config.ScenarioConfig())
    assert len(drawn) == 4
    assert {(scenario.beta, scenario.fleet_policy) for scenario in drawn} == {
        (0.02, "random"),
        (0.02, "cycle"),
        (0.1, "random"),
        (0.1, "cycle"),
    }
    assert len({scenario.seed for scenario in drawn}) == 4


def test_a_sweep_naming_an_unknown_parameter_is_refused(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"nonsense": [1, 2]}))
    with pytest.raises(ValueError, match="not a scenario parameter"):
        config.sweep_configs(sweep, config.ScenarioConfig())


# -- apportionment ---------------------------------------------------------


@pytest.mark.parametrize(
    ("weights", "total"),
    [
        ([1.0, 1.0, 1.0], 10),
        ([3.0, 2.0, 1.0], 7),
        ([0.0, 1.0], 5),
        ([0.0, 0.0], 4),
        ([1.0], 0),
        ([2.0, 2.0, 2.0, 2.0, 1.0], 999),
    ],
)
def test_largest_remainder_never_loses_a_passenger(weights: list[float], total: int) -> None:
    shares = allocate.largest_remainder(weights, total)
    assert sum(shares) == total
    assert all(share >= 0 for share in shares)


def test_largest_remainder_gives_the_leftover_to_the_largest_fraction() -> None:
    assert allocate.largest_remainder([1.0, 1.0, 1.0], 1, tie_break=[0.1, 0.9, 0.5]) == [0, 1, 0]


def test_the_vectorised_split_agrees_with_the_python_one() -> None:
    """The two implementations of the same method, on the same numbers.

    The jitter is generated once and handed to both, because the tie-break is
    part of the method: with the same draw for the same tie, the two have to
    agree seat for seat.
    """
    frame = pl.DataFrame(
        {
            "group": ["a"] * 5 + ["b"] * 5,
            "weight": [3.0, 1.0, 1.0, 0.5, 0.25, 2.0, 2.0, 1.0, 1.0, 0.1],
            "total": [97] * 5 + [13] * 5,
        }
    )
    split = allocate.apportion(
        frame, weights="weight", total="total", keys=["group"], seed=3, salt="t"
    )
    draws = allocate.jitter(3, "t", frame.height).to_list()
    for group in ("a", "b"):
        mine = split.filter(pl.col("group") == group)
        rows = frame.filter(pl.col("group") == group).height
        offset = 0 if group == "a" else rows
        reference = allocate.largest_remainder(
            mine["weight"].to_list(),
            int(mine["total"][0]),
            tie_break=draws[offset : offset + rows],
        )
        assert sum(mine["riders"].to_list()) == sum(reference) == int(mine["total"][0])
        assert sorted(mine["riders"].to_list(), reverse=True) == sorted(reference, reverse=True)


# -- the pair relation, against the reference implementation ----------------


#: The timetable columns the reference implementation reads.
REFERENCE_INPUTS = (
    "operator",
    "line",
    "direction",
    "service",
    "trip_id",
    "stop_seq",
    "stop_id",
    "stop",
    "minutes",
    "served",
)


def reference_pairs(frame: pl.DataFrame) -> pd.DataFrame:
    """`od.per_trip_pairs`, the pandas reference, on the same timetable.

    The hop goes through ``to_dicts`` rather than through Arrow: comparing the
    two does not justify a pyarrow dependency, and a list of tuples compares
    just as exactly as a frame does.
    """
    as_pandas = pd.DataFrame(frame.select(REFERENCE_INPUTS).to_dicts())
    as_pandas["minutes"] = as_pandas["minutes"].astype("float64")
    return od.per_trip_pairs(as_pandas)


def assert_same_pairs(mine: pl.DataFrame, reference: pd.DataFrame) -> None:
    """The same rows, whatever order either emitted them in."""
    keys = list(pairs.PAIR_COLUMNS)
    left = sorted(mine.select(keys).rows())
    right = sorted(tuple(row) for row in reference[keys].itertuples(index=False, name=None))
    assert len(left) == len(right), f"{len(left)} pairs against {len(right)}"
    assert left == right, "the polars pair relation disagrees with od.per_trip_pairs"


def test_the_pair_relation_matches_the_reference_on_the_toy_network(
    toy_timetable: pl.DataFrame,
) -> None:
    assert_same_pairs(pairs.per_trip_pairs(toy_timetable), reference_pairs(toy_timetable))


def test_a_stop_named_twice_in_one_minute_is_one_call(toy_timetable: pl.DataFrame) -> None:
    """Trip t3 calls at Alpha twice at 07:00; that is one call, not two."""
    mine = pairs.per_trip_pairs(toy_timetable).filter(pl.col("trip_id") == "t3")
    assert mine.height == 1
    assert mine["origin_id"][0] == "toy:alpha"
    assert mine["destination_id"][0] == "toy:beta"


def test_a_dashed_call_is_not_a_pair(toy_timetable: pl.DataFrame) -> None:
    """Trip t4 skips Gamma, so Beta->Alpha is a pair and Beta->Gamma is not."""
    mine = pairs.per_trip_pairs(toy_timetable.filter(pl.col("served"))).filter(
        pl.col("trip_id") == "t4"
    )
    assert set(mine["destination_id"].to_list()) == {"toy:alpha"}
    assert mine["run_min"][0] == 20


def test_pairs_carry_the_band_of_their_own_departure(toy_timetable: pl.DataFrame) -> None:
    """t5 leaves Beta at 09:20 and Gamma at 09:35: one trip, two bands."""
    stamped = pairs.with_time_keys(pairs.per_trip_pairs(toy_timetable))
    trip = stamped.filter(pl.col("trip_id") == "t5")
    assert dict(trip.group_by("origin_id").agg(pl.col("departure_band").unique()).iter_rows()) == {
        "toy:beta": ["am_peak"],
        "toy:gamma": ["midday"],
    }
    assert set(trip["service_group"].to_list()) == {"weekday"}


# -- inputs ----------------------------------------------------------------


def test_a_missing_supply_table_is_refused_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run `inmotion bus build` first"):
        inputs.read_network(tmp_path)


def test_the_timetable_is_read_only_where_the_trip_actually_calls(
    tmp_path: Path, toy_timetable: pl.DataFrame
) -> None:
    toy_timetable.write_csv(tmp_path / "stop_times.csv")
    stops = pl.DataFrame(
        {
            "stop_id": ["toy:alpha", "toy:beta", "toy:gamma"],
            "operator": ["toy"] * 3,
            "stop": ["Alpha", "Beta", "Gamma"],
            "trips": [3, 3, 2],
            "lines": ["1", "1", "1"],
        },
        schema={
            "stop_id": pl.String,
            "operator": pl.String,
            "stop": pl.String,
            "trips": pl.Int64,
            "lines": pl.String,
        },
    )
    stops.write_csv(tmp_path / "stops.csv")
    network = inputs.read_network(tmp_path)
    assert network.stop_times.height == toy_timetable.height
    assert network.served().height == 14
    assert network.served()["minutes"].dtype == pl.Int64
    trips = network.trips()
    assert trips.height == 5
    assert trips.filter(pl.col("trip_id") == "t1")["stops_served"][0] == 3
    assert trips.filter(pl.col("trip_id") == "t4")["departure_minutes"][0] == 720
    assert inputs.check(network, inputs.read_stop_attributes(None)) == [
        "1 timetable cells are dashes and carry no clock"
    ]


def test_stop_attributes_are_validated_at_the_door(tmp_path: Path) -> None:
    good = tmp_path / "stops.csv"
    good.write_text("stop_id,weight,category,lat,lon\ntoy:alpha,2.5,university,40.6,-8.6\n")
    frame = inputs.read_stop_attributes(good)
    assert frame.height == 1
    assert frame["weight"][0] == 2.5

    bad = tmp_path / "bad.csv"
    bad.write_text("stop_id,weight\ntoy:alpha,-1\n")
    with pytest.raises(ValueError, match="weight must be positive"):
        inputs.read_stop_attributes(bad)

    wrong_column = tmp_path / "wrong.csv"
    wrong_column.write_text("stop_id,attraction\ntoy:alpha,1\n")
    with pytest.raises(ValueError, match="does not read"):
        inputs.read_stop_attributes(wrong_column)

    duplicated = tmp_path / "twice.csv"
    duplicated.write_text("stop_id,lat\ntoy:alpha,40.6\ntoy:alpha,40.7\n")
    with pytest.raises(ValueError, match="names the same stop twice"):
        inputs.read_stop_attributes(duplicated)

    out_of_range = tmp_path / "range.csv"
    out_of_range.write_text("stop_id,lat\ntoy:alpha,140.6\n")
    with pytest.raises(ValueError, match="out of range"):
        inputs.read_stop_attributes(out_of_range)


@pytest.mark.skipif(
    not (paths.PROCESSED_DIR / "stop_times.csv").exists(),
    reason="bus tables not built (data/bus/processed is not tracked)",
)
def test_the_pair_relation_matches_the_reference_on_the_scraped_timetable() -> None:
    """The same parity, on all 24,000 rows the build actually produced."""
    served = inputs.read_network().served()
    assert_same_pairs(pairs.per_trip_pairs(served), reference_pairs(served))


# -- the demand model ------------------------------------------------------


def toy_network(tmp_path: Path, timetable: pl.DataFrame) -> inputs.Network:
    """Write a toy timetable and its stop dictionary where the loaders expect them."""
    stops = (
        timetable.filter(pl.col("served"))
        .group_by("stop_id", "operator", "stop")
        .agg(pl.col("trip_id").n_unique().alias("trips"))
        .with_columns(pl.lit("1").alias("lines"))
        .sort("stop_id")
    )
    timetable.write_csv(tmp_path / "stop_times.csv")
    stops.write_csv(tmp_path / "stops.csv")
    return inputs.read_network(tmp_path)


def demand_of(network: inputs.Network, config: config.ScenarioConfig) -> pl.DataFrame:
    """The generated demand for a toy network, through the same path build uses."""
    names = network.names()
    stamped = pairs.with_time_keys(pairs.name_stops(pairs.per_trip_pairs(network.served()), names))
    return model.generate(stamped, network, inputs.read_stop_attributes(None), config).pairs


def test_the_day_scales_with_the_service_it_runs(toy_timetable: pl.DataFrame) -> None:
    """Five trips at two boardings each is ten riders, whatever the pairs are."""
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        demand = demand_of(network, config.ScenarioConfig(boardings_per_departure=2.0))
        assert int(demand["demand"].sum()) == 10
        bigger = demand_of(network, config.ScenarioConfig(boardings_per_departure=25.0, seed=1))
        assert int(bigger["demand"].sum()) == 125


def test_an_explicit_total_is_split_over_the_network(toy_timetable: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        demand = demand_of(network, config.ScenarioConfig(daily_boardings=500))
        assert int(demand["demand"].sum()) == 500


def test_a_longer_ride_attracts_less(toy_timetable: pl.DataFrame) -> None:
    """With a steep decay the short pairs take almost all of the demand."""
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)

        def long_share(beta: float) -> float:
            demand = demand_of(
                network,
                config.ScenarioConfig(beta=beta, boardings_per_departure=100.0),
            )
            total = int(demand["demand"].sum())
            short = demand.filter(pl.col("min_run_min") <= 5)["demand"].sum()
            return short / total

        assert long_share(0.5) > long_share(0.01)


def test_a_manual_weight_moves_a_stop_up_the_ranking(toy_timetable: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        attributes = pl.DataFrame(
            {"stop_id": ["toy:gamma"], "weight": [50.0]},
            schema={"stop_id": pl.String, "weight": pl.Float64},
        )
        names = network.names()
        stamped = pairs.with_time_keys(
            pairs.name_stops(pairs.per_trip_pairs(network.served()), names)
        )
        plain = model.generate(
            stamped, network, inputs.read_stop_attributes(None), config.ScenarioConfig()
        ).pairs
        weighted = model.generate(stamped, network, attributes, config.ScenarioConfig()).pairs

        def into_gamma(frame: pl.DataFrame) -> int:
            return int(frame.filter(pl.col("destination_id") == "toy:gamma")["demand"].sum())

        assert into_gamma(weighted) > into_gamma(plain)


# -- the assignment and the fleet ------------------------------------------


def toy_assignment(toy_timetable: pl.DataFrame, policy: str, *, seed: int = 4):
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        names = network.names()
        stamped = pairs.with_time_keys(
            pairs.name_stops(pairs.per_trip_pairs(network.served()), names)
        )
        scenario = config.ScenarioConfig(
            seed=seed,
            fleet_policy=policy,
            boardings_per_departure=40.0,
            fleet=(
                config.BusSize("mini", 10, 1),
                config.BusSize("std", 40, 2),
                config.BusSize("bendy", 70, 1),
            ),
        )
        demand = model.generate(stamped, network, inputs.read_stop_attributes(None), scenario).pairs
        return assignment.assign(demand, stamped, scenario), demand


def test_the_assignment_keeps_every_rider(toy_timetable: pl.DataFrame) -> None:
    for policy in config.FLEET_POLICIES:
        assigned, demand = toy_assignment(toy_timetable, policy)
        assert int(assigned.intents["riders"].sum()) == int(demand["demand"].sum())


def test_every_trip_gets_a_bus_whichever_policy_ran(toy_timetable: pl.DataFrame) -> None:
    for policy in config.FLEET_POLICIES:
        assigned, _ = toy_assignment(toy_timetable, policy)
        assert assigned.fleet.height == 5
        assert set(assigned.fleet["bus_size"].unique().to_list()) <= {"mini", "std", "bendy"}


def test_the_fleet_cycles_in_departure_order(toy_timetable: pl.DataFrame) -> None:
    assigned, _ = toy_assignment(toy_timetable, "cycle")
    sizes = assigned.fleet.sort("trip_id")["bus_size"].to_list()
    assert len(set(sizes)) > 1


def test_the_biggest_buses_go_to_the_busiest_lines(toy_timetable: pl.DataFrame) -> None:
    """Every trip is on line 1 here, so the demand policy has one line to serve."""
    assigned, _ = toy_assignment(toy_timetable, "demand")
    assert set(assigned.fleet["bus_size"].to_list()) == {"bendy"}


def test_a_seeded_fleet_replays_and_a_new_seed_does_not(toy_timetable: pl.DataFrame) -> None:
    first, _ = toy_assignment(toy_timetable, "random", seed=4)
    again, _ = toy_assignment(toy_timetable, "random", seed=4)
    other, _ = toy_assignment(toy_timetable, "random", seed=9)
    assert first.fleet.equals(again.fleet)
    assert not first.fleet.equals(other.fleet)


# -- the queue -------------------------------------------------------------


def toy_flows(toy_timetable: pl.DataFrame, capacity: int, *, boardings: float = 30.0):
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        names = network.names()
        stamped = pairs.with_time_keys(
            pairs.name_stops(pairs.per_trip_pairs(network.served()), names)
        )
        scenario = config.ScenarioConfig(
            seed=2,
            fleet=(config.BusSize("tiny", capacity, 1),),
            fleet_policy="cycle",
            boardings_per_departure=boardings,
        )
        demand = model.generate(stamped, network, inputs.read_stop_attributes(None), scenario).pairs
        assigned = assignment.assign(demand, stamped, scenario)
        flows = run_day(network.served(), assigned.intents, assigned.fleet, scenario, names=names)
        return flows, demand, scenario


def test_a_bus_never_carries_more_than_it_has_seats(toy_timetable: pl.DataFrame) -> None:
    flows, _, _ = toy_flows(toy_timetable, capacity=8)
    trips = flows.trips
    assert trips.height
    assert (trips["onboard_after"] <= trips["capacity"]).all()
    assert (
        trips["onboard_arrival"] - trips["alighted"] + trips["boarded"] == trips["onboard_after"]
    ).all()
    assert (trips["boarded"] >= 0).all()
    assert (trips["denied"] >= 0).all()


def test_a_full_bus_leaves_riders_at_the_stop(toy_timetable: pl.DataFrame) -> None:
    """A one-seat bus on a busy line refuses riders, and the queue says so."""
    flows, _, _ = toy_flows(toy_timetable, capacity=1, boardings=200.0)
    assert int(flows.trips["denied"].sum()) > 0
    stops = flows.stops
    assert (stops["waiting_end"] >= 0).all()
    assert (
        stops["waiting_end"] == stops["waiting_start"] + stops["arrivals"] - stops["boarded"]
    ).all()


def test_the_day_accounts_for_every_rider_it_invented(toy_timetable: pl.DataFrame) -> None:
    flows, demand, _ = toy_flows(toy_timetable, capacity=6)
    assert int(flows.trips["boarded"].sum()) + flows.stranded == int(demand["demand"].sum())
    assert validate.check_flows(flows) == []


def test_the_same_seed_generates_the_same_day(toy_timetable: pl.DataFrame) -> None:
    first, _, _ = toy_flows(toy_timetable, capacity=6)
    again, _, _ = toy_flows(toy_timetable, capacity=6)
    assert first.trips.equals(again.trips)
    assert first.stops.equals(again.stops)
    assert first.od.equals(again.od)


def test_every_ride_that_was_boarded_is_in_the_od_relation(toy_timetable: pl.DataFrame) -> None:
    flows, _, _ = toy_flows(toy_timetable, capacity=6)
    assert int(flows.od["passengers"].sum()) == int(flows.trips["boarded"].sum())
    joins = flows.od.join(
        flows.trips.select("trip_id", "stop_id", "stop_seq"),
        left_on=["trip_id", "origin_id"],
        right_on=["trip_id", "stop_id"],
        how="left",
    )
    assert joins["stop_seq"].null_count() == 0


# -- matrices, instants and the run ----------------------------------------


def test_the_passenger_matrix_holds_the_flows_it_came_from(toy_timetable: pl.DataFrame) -> None:
    flows, _, _ = toy_flows(toy_timetable, capacity=6)
    matrix = matrices.passenger_matrix(flows.od)
    timeline = matrices.passenger_timeline(flows.od)
    assert int(matrix["passengers"].sum()) == int(flows.trips["boarded"].sum())
    assert validate.check_timeline(matrix, timeline) == []
    assert set(matrices.MATRIX_KEYS) <= set(matrix.columns)


def test_the_wide_grid_is_a_full_axis_with_zeros(toy_timetable: pl.DataFrame) -> None:
    flows, _, _ = toy_flows(toy_timetable, capacity=6)
    matrix = matrices.passenger_matrix(flows.od)
    axis = pl.DataFrame(
        {
            "operator": ["toy"] * 3,
            "origin_id": ["toy:alpha", "toy:beta", "toy:gamma"],
            "origin": ["Alpha", "Beta", "Gamma"],
        }
    )
    grid = matrices.wide(matrix, axis)
    assert grid.height == 3
    assert grid["toy:alpha"].sum() >= 0
    assert grid.null_count().sum_horizontal()[0] == 0


def test_the_instants_speak_the_label_vocabulary(toy_timetable: pl.DataFrame) -> None:
    flows, _, scenario = toy_flows(toy_timetable, capacity=6)
    table = instant_rows(flows.trips, scenario)
    assert table.height == flows.trips.height * 7  # -30s .. +30s every 10s
    visits = flows.trips.sort("trip_id", "minutes")
    at_visit = table.filter(pl.col("offset_s") == 0).sort("trip_id", "visit_minutes")
    assert at_visit["boarded"].to_list() == visits["boarded"].to_list()
    assert at_visit["alighted"].to_list() == visits["alighted"].to_list()
    assert at_visit["inside_bus"].to_list() == visits["onboard_before"].to_list()
    # Before the bus arrives there is nobody inside it, and the riders are still
    # at the stop.
    before = table.filter(pl.col("offset_s") < 0)
    assert (before["inside_bus"] == 0).all()
    assert (before["boarded"] == 0).all()
    assert (before["alighted"] == 0).all()
    after = table.filter(pl.col("offset_s") > 0)
    assert (after["boarded"] == 0).all()
    assert (table["at_stop"] >= 0).all()


def test_a_run_writes_a_scenario_that_reproduces(
    tmp_path: Path, toy_timetable: pl.DataFrame
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    toy_network(processed, toy_timetable)
    scenario = config.ScenarioConfig(
        seed=17,
        boardings_per_departure=20.0,
        fleet=(config.BusSize("std", 30, 2),),
    )
    first = build.run(scenario, source_dir=processed, out_dir=tmp_path / "one", with_map=False)
    second = build.run(scenario, source_dir=processed, out_dir=tmp_path / "two", with_map=False)
    assert first.files == second.files
    assert first.boardings == second.boardings
    assert first.problems == []
    assert (tmp_path / "one" / "flows.csv").read_text() == (
        tmp_path / "two" / "flows.csv"
    ).read_text()
    assert (tmp_path / "one" / "od_passengers.csv").read_text() == (
        tmp_path / "two" / "od_passengers.csv"
    ).read_text()
    assert (tmp_path / "one" / "scenario.json").exists()
    assert (tmp_path / "one" / "report.json").exists()
    record = json.loads((tmp_path / "one" / "scenario.json").read_text())
    assert record["seed"] == 17
    assert "stop_times.csv" in record["inputs"]

    other = build.run(
        config.ScenarioConfig(seed=18, boardings_per_departure=20.0),
        source_dir=processed,
        out_dir=tmp_path / "three",
        with_map=False,
    )
    assert other.fingerprint != first.fingerprint


def test_a_batch_writes_one_row_per_scenario(tmp_path: Path, toy_timetable: pl.DataFrame) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    toy_network(processed, toy_timetable)
    configs = [
        config.ScenarioConfig(seed=value, boardings_per_departure=10.0 + value) for value in (1, 2)
    ]
    reports = build.run_space(
        configs, source_dir=processed, out_root=tmp_path / "scenarios", with_map=False
    )
    assert len(reports) == 2
    index = pl.read_csv(tmp_path / "scenarios" / build.INDEX_FILE)
    assert index.height == 2
    assert set(index["boardings"].to_list()) == {report.boardings for report in reports}
    assert (tmp_path / "scenarios" / configs[0].resolved_label() / "flows.csv").exists()


# -- coordinates, without touching the network -----------------------------

#: A handful of OSM stops, shaped exactly like the ones Overpass returns.
OSM_FIXTURE = pl.DataFrame(
    {
        "osm_id": ["node/1", "node/2", "node/3", "node/4", "node/5", "node/6", "node/7"],
        "name": [
            "Barrocas",
            "Rua Clube dos Galitos",
            "Forca - Loja do Cidadão",
            "Quinta Clementina",
            "Avenida Cental",
            "Rua do Comércio",
            "Praça do Peixe",
        ],
        "lat": [40.6410, 40.6420, 40.6390, 40.6300, 40.6500, 40.6430, 40.6440],
        "lon": [-8.6500, -8.6530, -8.6560, -8.6400, -8.6600, -8.6540, -8.6520],
        "kind": ["bus_stop"] * 7,
    },
    schema={
        "osm_id": pl.String,
        "name": pl.String,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "kind": pl.String,
    },
)


def stops_to_place(names: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stop_id": [od.stop_id("toy", name) for name in names],
            "operator": ["toy"] * len(names),
            "stop": names,
        },
        schema={"stop_id": pl.String, "operator": pl.String, "stop": pl.String},
    )


def test_each_matching_tier_is_reachable_and_named() -> None:
    stops = stops_to_place(
        [
            "Rua Clube dos Galitos",  # exact
            "R. do Padrão Nova",  # expanded beyond the exact spelling
            "Barrocas A",  # the timetable's direction letter, OSM's plain name
            "Loja do Cidadão B",  # the name is contained in OSM's longer one
            "Avenida Central",  # one letter away from OSM's own spelling
            "Sítio Desconhecido",  # nowhere in the index
        ]
    )
    resolution = coords.match(stops, OSM_FIXTURE)
    placed = dict(resolution.frame.select("stop_id", "source").iter_rows())
    assert placed["toy:rua-clube-dos-galitos"] == "exact"
    assert placed["toy:barrocas-a"] == "tokens"
    assert placed["toy:loja-do-cidadao-b"] == "tokens"
    assert placed["toy:avenida-central"] == "fuzzy"
    assert placed["toy:sitio-desconhecido"] == "none"


def test_the_abbreviations_a_timetable_prints_are_spelled_out() -> None:
    stops = stops_to_place(["Qt? Clementina", "Av. da Força Aérea"])
    index = pl.concat(
        [
            OSM_FIXTURE,
            pl.DataFrame(
                {
                    "osm_id": ["node/8", "node/9"],
                    "name": ["Quinta Clementina", "Avenida da Força Aérea"],
                    "lat": [40.63, 40.65],
                    "lon": [-8.64, -8.66],
                    "kind": ["bus_stop", "bus_stop"],
                },
                schema=OSM_FIXTURE.schema,
            ),
        ]
    )
    placed = dict(coords.match(stops, index).frame.select("stop_id", "source").iter_rows())
    assert placed["toy:qt-clementina"] == "expanded"
    assert placed["toy:av-da-forca-aerea"] == "expanded"


def test_an_unplaced_stop_is_left_blank_rather_than_guessed() -> None:
    resolution = coords.match(stops_to_place(["Nenhum Lugar Assim"]), OSM_FIXTURE)
    row = resolution.frame.row(0, named=True)
    assert row["source"] == "none"
    assert row["lat"] is None
    assert row["lon"] is None
    assert any("no coordinates" in warning for warning in resolution.warnings)


def test_a_given_coordinate_beats_any_match() -> None:
    stops = stops_to_place(["Barrocas A"])
    manual = pl.DataFrame(
        {"stop_id": ["toy:barrocas-a"], "lat": [41.0], "lon": [-8.0]},
        schema={"stop_id": pl.String, "lat": pl.Float64, "lon": pl.Float64},
    )
    row = coords.match(stops, OSM_FIXTURE, manual=manual).frame.row(0, named=True)
    assert row["source"] == "manual"
    assert row["lat"] == 41.0


def test_the_two_sides_of_one_place_share_a_position() -> None:
    """Whichever side matches first fixes the other, with no OSM stop of its own."""
    stops = stops_to_place(["Barrocas A", "Barrocas B"])
    index = OSM_FIXTURE.filter(pl.col("name") == "Praça do Peixe")
    resolution = coords.match(
        stops,
        pl.concat(
            [
                index,
                pl.DataFrame(
                    {
                        "osm_id": ["node/1"],
                        "name": ["Barrocas"],
                        "lat": [40.641],
                        "lon": [-8.651],
                        "kind": ["bus_stop"],
                    },
                    schema=OSM_FIXTURE.schema,
                ),
            ]
        ),
    )
    rows = {row["stop_id"]: row for row in resolution.frame.iter_rows(named=True)}
    assert rows["toy:barrocas-a"]["lat"] == rows["toy:barrocas-b"]["lat"] is not None


def test_stops_close_enough_to_be_one_place_get_one_cluster() -> None:
    frame = pl.DataFrame(
        {
            "stop_id": ["a", "b", "c"],
            "lat": [40.6400, 40.6402, 40.7000],  # ~22 m apart, then 6 km away
            "lon": [-8.6500, -8.6500, -8.6500],
            "source": ["exact", "exact", "exact"],
            "osm_id": ["", "", ""],
            "score": [1.0, 1.0, 1.0],
        }
    )
    clustered = coords.cluster(frame)
    ids = clustered["cluster_id"].to_list()
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_the_coordinate_tables_are_written_with_the_gaps_named(tmp_path: Path) -> None:
    stops = stops_to_place(["Barrocas A", "Sítio Desconhecido"])
    resolution = coords.match(stops, OSM_FIXTURE)
    enriched = resolution.frame.with_columns(pl.lit("toy").alias("operator"))
    written = coords.write(enriched, tmp_path)
    assert {path.name for path in written} == {
        coords.COORDS_FILE,
        coords.CLUSTERS_FILE,
        coords.UNRESOLVED_FILE,
    }
    unresolved = pl.read_csv(tmp_path / coords.UNRESOLVED_FILE)
    assert unresolved["stop_id"].to_list() == ["toy:sitio-desconhecido"]
    assert pl.read_csv(tmp_path / coords.COORDS_FILE).height == 2


def test_an_osm_index_that_answers_nothing_is_reported_not_ignored() -> None:
    resolution = coords.match(
        stops_to_place(["Barrocas A"]), pl.DataFrame(schema=OSM_FIXTURE.schema)
    )
    assert resolution.counts() == {"none": 1}
    assert any("index is empty" in warning for warning in resolution.warnings)


# -- the results document, held to the tables ------------------------------

DOCUMENT = paths.REPO_ROOT / "AI_RESULT.md"

requires_results = pytest.mark.skipif(
    not (DOCUMENT.exists() and paths.SCENARIO_DIR.exists()),
    reason="no generated scenarios or no results document to hold them to",
)


def documented(pattern: str) -> str:
    """The number the document states, refusing to pass when it stopped doing so."""
    match = re.search(pattern, DOCUMENT.read_text())
    assert match is not None, f"AI_RESULT.md no longer states anything matching {pattern!r}"
    return match.group(1).replace(",", "")


def documented_scenario() -> Path:
    """The scenario the results section is written about."""
    label = re.search(r"For `(s[\w.\-]+)`:", DOCUMENT.read_text())
    assert label is not None, "AI_RESULT.md no longer names the scenario it describes"
    directory = paths.SCENARIO_DIR / label.group(1)
    if not directory.exists():
        pytest.skip(f"{directory} has not been generated in this checkout")
    return directory


@requires_results
def test_the_document_states_the_row_counts_of_the_generated_tables() -> None:
    directory = documented_scenario()
    stated = {
        "flows.csv": documented(r"\| `flows\.csv` \| ([\d,]+) \|"),
        "stop_flow.csv": documented(r"\| `stop_flow\.csv` \| ([\d,]+) \|"),
        "od_flow.csv": documented(r"\| `od_flow\.csv` \| ([\d,]+) \|"),
        "od_demand.csv": documented(r"\| `od_demand\.csv` \| ([\d,]+) \|"),
        "od_passengers.csv": documented(r"\| `od_passengers\.csv` \| ([\d,]+) \|"),
        "od_passengers_timeline.csv": documented(
            r"\| `od_passengers_timeline\.csv` \| ([\d,]+) \|"
        ),
        "instants.csv": documented(r"\| `instants\.csv` \| ([\d,]+) \|"),
    }
    actual = {
        name: len(pl.read_csv(directory / name, schema_overrides={"line": pl.String}))
        for name in stated
    }
    assert {name: int(value) for name, value in stated.items()} == actual


@requires_results
def test_the_document_states_the_totals_the_generator_reported() -> None:
    directory = documented_scenario()
    report = json.loads((directory / "report.json").read_text())
    stated = {
        "demand": documented(r"stop pairs\s+[\d,]+\s+demand\s+([\d,]+)"),
        "boarded": documented(r"boarded\s+([\d,]+)\s+alighted"),
        "alighted": documented(r"alighted\s+([\d,]+)"),
        "never boarded": documented(r"never boarded\s+([\d,]+)"),
        "matrix cells": documented(r"matrix cells\s+([\d,]+)"),
    }
    assert int(stated["demand"]) == report["demand"]
    assert int(stated["boarded"]) == report["boardings"]
    assert int(stated["alighted"]) == report["alightings"]
    assert int(stated["never boarded"]) == report["stranded"]
    assert int(stated["matrix cells"]) == report["matrix_cells"]
    assert report["problems"] == [], "the documented scenario failed its own invariants"


@requires_results
def test_the_document_states_the_batch_it_printed() -> None:
    index = pl.read_csv(paths.SCENARIO_DIR / build.INDEX_FILE)
    pattern = (
        r"\| (\d+) \| (random|cycle|demand) \| ([\d.]+) \| ([\d,]+) \|"
        r" ([\d,]+) \| ([\d,]+) \| (\d+) \|"
    )
    rows = re.findall(pattern, DOCUMENT.read_text())
    assert len(rows) == 3, "the batch table in AI_RESULT.md lost a row or changed shape"
    for seed, policy, per_departure, demand, boarded, stranded, problems in rows:
        row = index.filter(pl.col("seed") == int(seed)).row(0, named=True)
        assert row["fleet_policy"] == policy
        assert float(per_departure) == pytest.approx(row["boardings_per_departure"], abs=0.05)
        assert int(demand.replace(",", "")) == row["demand"]
        assert int(boarded.replace(",", "")) == row["boardings"]
        assert int(stranded.replace(",", "")) == row["stranded"]
        assert int(problems) == row["problems"]


@requires_results
def test_the_document_states_the_coordinate_rate_it_achieved() -> None:
    table = pl.read_csv(
        paths.processed(coords.COORDS_FILE), schema_overrides={"stop_id": pl.String}
    )
    placed = table.filter(pl.col("source") != "none").height
    stated = re.search(
        r"\| \*\*placed\*\* \| \*\*([\d,]+) of ([\d,]+)\*\* \| \*\*([\d.]+) %\*\* \|",
        DOCUMENT.read_text(),
    )
    assert stated is not None, "AI_RESULT.md no longer states how many stops were placed"
    assert int(stated.group(1).replace(",", "")) == placed
    assert int(stated.group(2).replace(",", "")) == table.height
    assert float(stated.group(3)) == pytest.approx(100 * placed / table.height, abs=0.05)
    tiers = dict(table.group_by("source").len().iter_rows())
    for source, count in (("tokens", 884), ("exact", 467), ("fuzzy", 25), ("expanded", 11)):
        assert tiers.get(source) == count, f"{source}: {tiers.get(source)} against {count} stated"


# -- the page: the basemap it can and cannot draw ---------------------------


def test_every_basemap_a_scenario_allows_is_one_the_page_knows() -> None:
    """Two tables, one allowed set; this is what keeps them from drifting."""
    assert set(config.BASEMAPS) == set(map_module.BASE_LAYERS)
    assert config.DEFAULT_BASEMAP in map_module.BASE_LAYERS


def map_page(*, basemap: str | None = None, legs=None, flows=None, lines=None) -> str:
    """One map page, built the way `map.write` builds it, with no data needed."""
    chosen = basemap or config.DEFAULT_BASEMAP
    pages_list = map_module.basemap_payload()
    operators = sorted({feature["properties"]["operator"] for feature in (flows or [])}) or ["toy"]
    groups = sorted({feature["properties"]["service_group"] for feature in (flows or [])}) or [
        "weekday"
    ]
    return pages_module.render_map(
        [],
        flows or [],
        legs or [],
        basemaps=pages_list,
        chosen=next(entry for entry in pages_list if entry["name"] == chosen),
        hours={"min": 0, "max": 24, "labels": [f"{hour:02d}:00" for hour in range(25)]},
        lines=lines or [],
        operators=operators,
        groups=groups,
        config=config.ScenarioConfig(basemap=chosen),
        report=build.Report(label="toy", fleet="std (40 seats) x 1 trips"),
        matrix=pl.DataFrame(),
    )


def test_the_page_draws_the_chosen_basemap_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.CARTO_KEY_ENV, "test-key-not-a-real-secret")
    for name, layer in map_module.BASE_LAYERS.items():
        page = map_page(basemap=name)
        assert f'"name":"{name}"' in page
        if layer["url"]:
            template = layer["url"].replace("{key}", "test-key-not-a-real-secret")
            assert template in page, f"{name} is not drawn"
        else:
            # No tiles: nothing to 403 and nothing to authorise.
            assert "tileLayer(" in page
            assert '"url":""' in page


def test_the_default_page_reaches_for_no_third_party_tiles_at_all() -> None:
    """The regression this exists for, twice over.

    OpenStreetMap's tile servers answer a ``file://`` page with 403 (no Referer),
    and CARTO's basemap CDN now answers an API-key demand. A page opened from
    disk must therefore draw something nobody can take away: the network's own
    legs, which come from the timetable.
    """
    legs = [[[-8.65, 40.64], [-8.66, 40.65]]]
    page = map_page(legs=legs)
    # The menu offers every basemap so the viewer can switch; what must not
    # happen is that the *drawn* one reaches for a third party.
    assert '"basemap":{"name":"network","url":""' in page
    assert "Basemap: network" in page
    assert '"legs":[[[-8.65,40.64],[-8.66,40.65]]]' in page
    # And the switcher refuses to draw a layer it has no URL for.
    assert "if (!chosen.available || !chosen.url) { return; }" in page


def test_the_osm_basemap_warns_that_it_needs_http() -> None:
    """Offered for a served page, and honest about why it cannot work from disk."""
    page = map_page(basemap="osm")
    assert "tile.openstreetmap.org" in page
    assert "python3 -m http.server" in page
    assert "Basemap: osm" in page


def test_an_unknown_basemap_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown basemap"):
        config.ScenarioConfig(basemap="terrain")


# -- the basemap key: read from the environment, never from the scenario -----


def test_auto_basemap_follows_whether_a_key_is_configured() -> None:
    assert config.resolve_basemap("auto", {config.CARTO_KEY_ENV: "a-key"}) == "carto-dark"
    assert config.resolve_basemap("auto", {}) == "network"
    assert config.resolve_basemap("none", {config.CARTO_KEY_ENV: "a-key"}) == "none"


def test_a_keyed_basemap_without_a_key_is_refused_with_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(config.CARTO_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=config.CARTO_KEY_ENV):
        map_module.require_key("carto-dark", {})
    map_module.require_key("network", {})  # needs nothing


def test_the_key_reaches_the_tiles_and_the_scenario_never_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page must carry the key to draw; the record must not.

    A client-side basemap key is visible in the page by construction. What has to
    hold is that it is not written into `scenario.json`, which is the file people
    commit and diff — so the scenario records the basemap by name only.
    """
    secret = "test-key-not-a-real-secret"
    monkeypatch.setenv(config.CARTO_KEY_ENV, secret)
    page = map_page(basemap="carto-dark")
    assert f"?key={secret}" in page
    assert "basemaps.cartocdn.com/dark_all" in page
    record = config.ScenarioConfig(basemap="carto-dark").to_dict()
    assert record["basemap"] == "carto-dark"
    assert secret not in json.dumps(record)


def test_a_run_that_cannot_draw_its_map_refuses_before_writing(
    tmp_path: Path, toy_timetable: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    toy_network(processed, toy_timetable)
    settled = tmp_path / "scenario"
    monkeypatch.delenv(config.CARTO_KEY_ENV, raising=False)
    (processed / coords.COORDS_FILE).write_text(
        "stop_id,lat,lon,source,osm_id,score,cluster_id,operator\n"
        "toy:alpha,40.64,-8.65,exact,,1.0,0,toy\n"
    )
    with pytest.raises(RuntimeError, match=config.CARTO_KEY_ENV):
        build.run(
            config.ScenarioConfig(basemap="carto-dark"),
            source_dir=processed,
            out_dir=settled,
            with_map=True,
        )
    assert not (settled / "flows.csv").exists(), "a scenario was left half-written"


# -- the viewer: filtering, time, and the space-time cube ------------------


def test_a_flow_carries_the_lines_and_the_hourly_shape_the_filters_need() -> None:
    """Filtering happens in the page, so the payload has to make it possible.

    A relation that only says how many riders it carried over the day cannot be
    filtered by line, or played through the hours; these two fields are what the
    viewer's controls read.
    """
    od = pl.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3"],
            "operator": ["toy"] * 3,
            "line": ["7", "7", "9"],
            "direction": ["a"] * 3,
            "service_group": ["weekday"] * 3,
            "origin_id": ["toy:a"] * 3,
            "origin": ["Alpha"] * 3,
            "destination_id": ["toy:b"] * 3,
            "destination": ["Beta"] * 3,
            "departure_minutes": [7 * 60, 7 * 60 + 20, 8 * 60],
            "departure_band": ["am_peak"] * 3,
            "run_min": [10, 10, 10],
            "arrival_minutes": [7 * 60 + 10, 7 * 60 + 30, 8 * 60 + 10],
            "passengers": [3, 2, 5],
        },
        schema_overrides={"line": pl.String, "operator": pl.String, "trip_id": pl.String},
    )
    coords = pl.DataFrame(
        {
            "stop_id": ["toy:a", "toy:b"],
            "lat": [40.64, 40.65],
            "lon": [-8.65, -8.66],
            "operator": ["toy"] * 2,
        }
    )
    features = map_module.flow_features(od, coords)
    assert len(features) == 1
    properties = features[0]["properties"]
    assert properties["lines"] == "7 9"
    assert properties["hours"] == [[7, 5], [8, 5]]
    assert properties["passengers"] == 10


def test_the_cube_reads_the_same_ride_twice_by_its_two_clocks() -> None:
    """Entries belong to the hour a rider stepped on, exits to the one they left."""
    od = pl.DataFrame(
        {
            "trip_id": ["t1"],
            "operator": ["toy"],
            "line": ["7"],
            "direction": ["a"],
            "service_group": ["weekday"],
            "origin_id": ["toy:a"],
            "origin": ["Alpha"],
            "destination_id": ["toy:b"],
            "destination": ["Beta"],
            "departure_minutes": [7 * 60 + 55],
            "departure_band": ["am_peak"],
            "run_min": [15],
            "arrival_minutes": [8 * 60 + 10],
            "passengers": [4],
        },
        schema_overrides={"line": pl.String, "operator": pl.String, "trip_id": pl.String},
    )
    flows = pl.DataFrame(
        {
            "stop_id": ["toy:a", "toy:b"],
            "boarded": [4, 0],
            "alighted": [0, 4],
            "waiting_at_arrival": [0, 0],
        },
        schema_overrides={"stop_id": pl.String},
    )
    stops = pl.DataFrame(
        {
            "stop_id": ["toy:a", "toy:b"],
            "stop": ["Alpha", "Beta"],
            "operator": ["toy", "toy"],
            "lines": ["7", "7"],
        }
    )
    coords = pl.DataFrame(
        {
            "stop_id": ["toy:a", "toy:b"],
            "lat": [40.64, 40.65],
            "lon": [-8.65, -8.66],
        }
    )
    cube = map_module.cube_payload(od, flows, stops, coords)
    assert len(cube["points"]) == 1
    point = cube["points"][0]
    assert point["b"] == 7, "the ride boards at 07:55"
    assert point["a"] == 8, "and alights at 08:10, in the next hour"
    assert point["n"] == 4


def test_the_cube_only_holds_the_stops_it_says_it_holds() -> None:
    od = pl.read_csv(
        paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend" / "od_flow.csv",
        schema_overrides={"line": pl.String},
    )
    flows = pl.read_csv(
        paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend" / "flows.csv",
        schema_overrides={"line": pl.String},
    )
    network = inputs.read_network()
    positions = pl.read_csv(
        paths.processed(coords.COORDS_FILE), schema_overrides={"stop_id": pl.String}
    )
    cube = map_module.cube_payload(od, flows, network.dictionary(), positions, stops_limit=25)
    assert len(cube["stops"]) == 25
    assert cube["omitted"] > 0, "the payload has to say how much it left out"
    indexes = set(range(25))
    assert all(point["o"] in indexes and point["d"] in indexes for point in cube["points"])


def test_the_pages_offer_the_filters_and_the_time_controls() -> None:
    """The controls exist in the markup, and the payload gives them something to read."""
    page = map_page(
        flows=[
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-8.65, 40.64], [-8.66, 40.65]]},
                "properties": {
                    "from": "Alpha",
                    "to": "Beta",
                    "origin_id": "toy:a",
                    "destination_id": "toy:b",
                    "service_group": "weekday",
                    "operator": "toy",
                    "lines": "7",
                    "hours": [[7, 3]],
                    "passengers": 3,
                },
            }
        ],
        lines=["7"],
    )
    for control in (
        'id="basemap"',
        'id="operator-row"',
        'id="line-row"',
        'id="min-passengers"',
        'id="stop-search"',
        'id="hour-from"',
        'id="hour-to"',
        'id="play"',
        'id="sparkline"',
    ):
        assert control in page, f"the viewer lost its {control} control"
    assert '"lines":["7"]' in page
    assert '"hours":{"min":0,"max":24' in page


def test_the_cube_page_offers_entries_exits_time_and_routes() -> None:
    od = pl.read_csv(
        paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend" / "od_flow.csv",
        schema_overrides={"line": pl.String},
    )
    flows = pl.read_csv(
        paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend" / "flows.csv",
        schema_overrides={"line": pl.String},
    )
    network = inputs.read_network()
    positions = pl.read_csv(
        paths.processed(coords.COORDS_FILE), schema_overrides={"stop_id": pl.String}
    )
    cube = map_module.cube_payload(od, flows, network.dictionary(), positions)
    hours = map_module.hour_window(od)
    page = pages_module.render_cube(
        cube,
        config.ScenarioConfig(),
        build.Report(label="toy"),
        hours=hours,
        shifted=1,
        stop_total=network.dictionary().height,
    )
    for control in (
        'name="mode" value="entries"',
        'name="mode" value="exits"',
        'id="hour-from"',
        'id="hour-to"',
        'id="play"',
        'id="line-row"',
        'id="toggle-all"',
    ):
        assert control in page, f"the cube lost its {control} control"
    assert '"points":[{' in page
    assert "OrbitControls" in page
    assert sorted(set(re.findall(r"__[A-Z_]+__", page))) == []


def test_the_cube_lattice_says_which_hours_it_spans() -> None:
    od = pl.read_csv(
        paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend" / "od_flow.csv",
        schema_overrides={"line": pl.String},
    )
    window = map_module.hour_window(od)
    assert window["min"] <= window["max"]
    assert len(window["labels"]) == window["max"] + 1
    assert window["labels"][7] == "07:00"


# -- the viewer's client-side logic, actually executed ---------------------

NODE = shutil.which("node")

requires_node = pytest.mark.skipif(
    NODE is None, reason="no node on PATH to execute the page's filter logic with"
)


def run_javascript(source: str) -> dict:
    """Evaluate a snippet in node and return the JSON it printed."""
    script = source + "\nconsole.log(JSON.stringify(RESULT));\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(script)
        path = handle.name
    try:
        finished = subprocess.run(
            [NODE, path], capture_output=True, text=True, timeout=60, check=False
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert finished.returncode == 0, finished.stderr.strip()[:800]
    return json.loads(finished.stdout.strip().splitlines()[-1])


@requires_node
def test_the_filter_logic_executes_and_filters_by_line_and_hour() -> None:
    """The page's own filter code, run for real rather than trusted.

    The lines and the hourly slots are the two dimensions a reader filters on, and
    both are pure functions over the payload — so they can be executed here
    instead of being taken on faith until somebody opens the page.
    """
    harness = f"""
const DATA = {{
  groups: ['weekday'], operators: ['toy'], lines: ['7', '9'],
  hours: {{min: 6, max: 22}}
}};
{pages_module.FILTER_JS}
const state = {{weekday: true, toy: true, lines: {{ '7': true, '9': true }}, allLines: true}};
const flow = {{properties: {{lines: '7 9', service_group: 'weekday', operator: 'toy'}}}};
const onlyNine = {{properties: {{lines: '9', service_group: 'weekday', operator: 'toy'}}}};
const window = hourWindow(DATA, {{from: 7, to: 8}});
const RESULT = {{
  window: window,
  inside: [inWindow(7, window), inWindow(8, window), inWindow(9, window)],
  matchesBoth: lineMatches(DATA, state, flow),
  allOn: allOn(DATA, state),
  filtered: (function () {{
    state.lines['7'] = false; state.allLines = false;
    return [lineMatches(DATA, state, flow), lineMatches(DATA, state, onlyNine)];
  }})(),
  singleLine: (function () {{
    const bar = {{properties: {{lines: '7'}}}};
    state.lines['7'] = false;
    return lineMatches(DATA, state, bar);
  }})()
}};
"""
    result = run_javascript(harness)
    assert result["window"] == {"from": 7, "to": 8, "span": 17}
    assert result["inside"] == [True, True, False]
    assert result["matchesBoth"] is True
    assert result["allOn"] is True
    assert result["filtered"] == [True, True], "a two-line cell survives hiding one of them"
    assert result["singleLine"] is False, "a cell whose only line is hidden disappears"


@requires_node
def test_the_hourly_riders_are_summed_within_the_window_only() -> None:
    harness = f"""
const DATA = {{groups: [], operators: [], lines: [], hours: {{min: 0, max: 24}}}};
{pages_module.FILTER_JS}
const feature = {{properties: {{hours: [[7, 3], [8, 4], [12, 10]]}}}};
const RESULT = {{
  morning: hourRiders(feature, {{from: 7, to: 8}}),
  midday: hourRiders(feature, {{from: 12, to: 12}}),
  empty: hourRiders(feature, {{from: 20, to: 22}}),
  none: hourRiders({{properties: {{}}}}, {{from: 0, to: 24}})
}};
"""
    result = run_javascript(harness)
    assert result == {"morning": 7, "midday": 10, "empty": 0, "none": 0}


# -- the pages, rendered by a real browser --------------------------------

CHROME = next(
    (
        candidate
        for candidate in (
            shutil.which("google-chrome-stable"),
            shutil.which("google-chrome"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
        )
        if candidate
    ),
    None,
)


def cdn_answers() -> bool:
    """Whether the pages' two CDN dependencies can be reached at all."""
    try:
        request = urllib.request.Request(
            "https://unpkg.com/three@0.147.0/build/three.min.js",
            method="HEAD",
            headers={"User-Agent": "inmotion-tests/0.1"},
        )
        with urllib.request.urlopen(request, timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


requires_browser = pytest.mark.skipif(
    CHROME is None or not cdn_answers(),
    reason="needs a headless browser and the CDN the pages load three.js and Leaflet from",
)


def screenshot(page: Path, output: Path, *, budget_ms: int = 7000) -> str:
    """Render a page in headless Chrome and return everything it logged."""
    finished = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
            "--hide-scrollbars",
            "--window-size=1200,800",
            f"--virtual-time-budget={budget_ms}",
            "--enable-logging=stderr",
            "--v=0",
            f"--screenshot={output}",
            page.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return finished.stderr


def coloured_pixels(png: Path, *, panel_px: int = 400) -> int:
    """Count the canvas pixels that carry real colour.

    Greys will not do: the panel's text and the lattice are grey, and counting
    those let a scene of black cubes pass as rendered. The flows are the only
    saturated thing on the canvas, so saturation is what has to be present. The
    panel is cropped off because it draws itself whether or not the canvas ever
    worked — which is exactly how a black cube once passed for a working page.
    """
    with Image.open(png) as image:
        canvas = image.convert("RGB")
        width, height = canvas.size
        region = canvas.crop((panel_px, 0, width, height))
        coloured = 0
        for red, green, blue in region.getdata():
            if max(red, green, blue) - min(red, green, blue) > 40:
                coloured += 1
        return coloured


def write_showcase(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny pair of pages, rendered the way the generator renders them."""
    stops = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-8.65 + index * 0.01, 40.64]},
            "properties": {
                "stop_id": f"toy:{index}",
                "name": f"Stop {index}",
                "operator": "toy",
                "lines": "7",
                "boarded": 10 - index,
                "alighted": 4,
                "denied": index,
                "peak_queue": 3,
                "cluster_id": index,
                "source": "exact",
            },
        }
        for index in range(6)
    ]
    flows = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[-8.65 + a * 0.01, 40.64], [-8.65 + b * 0.01, 40.64]],
            },
            "properties": {
                "from": f"Stop {a}",
                "to": f"Stop {b}",
                "origin_id": f"toy:{a}",
                "destination_id": f"toy:{b}",
                "service_group": "weekday",
                "operator": "toy",
                "lines": "7",
                "hours": [[8, 3], [17, 5]],
                "passengers": 8,
            },
        }
        for a, b in ((0, 3), (1, 4), (2, 5))
    ]
    legs = [
        [feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][1]]
        for feature in flows
    ]
    report = build.Report(
        label="showcase", boardings=48, alightings=24, fleet="std (40 seats) x 2 trips"
    )
    map_page_html = pages_module.render_map(
        stops,
        flows,
        legs,
        basemaps=map_module.basemap_payload({"CARTO_API_KEY": ""}),
        chosen=next(
            entry
            for entry in map_module.basemap_payload({"CARTO_API_KEY": ""})
            if entry["name"] == "network"
        ),
        hours={"min": 8, "max": 17, "labels": [f"{hour:02d}:00" for hour in range(18)]},
        lines=["7"],
        operators=["toy"],
        groups=["weekday"],
        config=config.ScenarioConfig(),
        report=report,
        matrix=pl.DataFrame(),
    )
    cube_points = [
        {
            "o": index % 6,
            "d": (index * 3 + 1) % 6,
            "b": 8 + index % 6,
            "a": 9 + index % 5,
            "n": 1 + index % 4,
            "l": 0,
            "p": 0,
            "s": 0,
        }
        for index in range(40)
    ]
    cube = {
        "stops": [
            {
                "id": f"toy:{index}",
                "name": f"Stop {index}",
                "operator": "toy",
                "lines": "7",
                "boarded": 10,
                "alighted": 4,
                "peak_queue": 3,
                "lat": 40.64,
                "lon": -8.65,
            }
            for index in range(6)
        ],
        "points": cube_points,
        "lines": ["7"],
        "operators": ["toy"],
        "groups": ["weekday"],
        "omitted": 0,
    }
    cube_page_html = pages_module.render_cube(
        cube,
        config.ScenarioConfig(),
        report,
        hours={"min": 8, "max": 17, "labels": [f"{hour:02d}:00" for hour in range(18)]},
        shifted=5,
        stop_total=6,
    )
    replay_html = replay_page_for_test()
    map_path = tmp_path / "map.html"
    cube_path = tmp_path / "cube.html"
    replay_path = tmp_path / "replay.html"
    map_path.write_text(map_page_html)
    cube_path.write_text(cube_page_html)
    replay_path.write_text(replay_html)
    return map_path, cube_path, replay_path


@requires_browser
@pytest.mark.parametrize("page", ["map", "cube", "replay"])
def test_the_pages_render_without_a_script_error(tmp_path: Path, page: str) -> None:
    """The page has to survive its own first frame in a browser, not just parse.

    Syntax checks cannot see a helper that only the map page defined, a material
    that renders black, or a camera fitted before the canvas had a size — all
    three have broken this viewer at least once.
    """
    map_page_path, cube_page_path, replay_path = write_showcase(tmp_path)
    source = {"map": map_page_path, "cube": cube_page_path, "replay": replay_path}[page]
    shot = tmp_path / f"{page}.png"
    log = screenshot(source, shot)
    for fatal in ("Uncaught ", "ReferenceError", "TypeError", "SyntaxError"):
        assert fatal not in log, [line for line in log.splitlines() if fatal in line][:3]
    assert shot.exists(), "the browser wrote no screenshot"
    assert coloured_pixels(shot) > 500, f"the {page} canvas drew nothing recognisable"


# -- the route replay ------------------------------------------------------


def replay_of(scenario: str) -> tuple[dict, dict]:
    """The replay payload and the flow table of one generated scenario."""
    root = paths.SCENARIO_DIR / scenario
    flows = pl.read_csv(root / "flows.csv", schema_overrides={"line": pl.String})
    positions = pl.read_csv(
        paths.processed(coords.COORDS_FILE), schema_overrides={"stop_id": pl.String}
    )
    network = inputs.read_network()
    return map_module.replay_payload(flows, network.dictionary(), positions), {"flows": flows}


def test_the_replay_holds_only_trips_it_can_draw() -> None:
    """A trip that calls at a stop with no position cannot be replayed.

    A bus that disappears for two stops and reappears somewhere else would be a
    claim about the geometry that the data does not support, so those trips are
    left out — and counted, so the page can say so.
    """
    payload, _ = replay_of("s0042-random-bpd12-weekday-weekend")
    assert payload["routes"], "no route can be replayed"
    assert payload["omitted"] > 0
    placed = {stop["id"] for stop in payload["stops"]}
    assert payload["stops"]
    assert all(stop["lat"] and stop["lon"] for stop in payload["stops"])
    for route in payload["routes"]:
        assert len(route["trips"]) >= 1
        assert len(route["trips"]) <= route["trips_all"]
        assert len(route["shape"]) == max(len(trip["events"]) for trip in route["trips"])
        for trip in route["trips"]:
            assert trip["capacity"] > 0
            # every event is a stop the page has a position for, in time order
            for event in trip["events"]:
                assert payload["stops"][event[0]]["id"] in placed
            minutes = [event[1] for event in trip["events"]]
            assert minutes == sorted(minutes)
            onboard = [event[4] for event in trip["events"]]
            assert all(0 <= value <= trip["capacity"] for value in onboard)
    # the busiest route first: the page opens on the one worth watching
    boarded = [route["boarded"] for route in payload["routes"]]
    assert boarded == sorted(boarded, reverse=True)
    assert placed


def test_the_replay_carries_the_numbers_the_page_prints() -> None:
    """Every number on the panel comes from this payload, so they must agree."""
    payload, tables = replay_of("s0042-random-bpd12-weekday-weekend")
    route = payload["routes"][0]
    flows = tables["flows"]
    mine = flows.filter(
        (pl.col("operator") == route["operator"])
        & (pl.col("line") == route["line"])
        & (pl.col("direction") == route["direction"])
        & (pl.col("service_group") == route["service"])
    )
    replayed = {trip["id"] for trip in route["trips"]}
    assert route["boarded"] == int(mine.filter(pl.col("trip_id").is_in(replayed))["boarded"].sum())
    assert route["trips_all"] == mine["trip_id"].n_unique()
    assert route["day"]["from"] == int(
        mine.filter(pl.col("trip_id").is_in(replayed))["minutes"].min()
    )


def test_the_replay_clock_runs_from_the_first_bus_to_the_last() -> None:
    payload, _ = replay_of("s0042-random-bpd12-weekday-weekend")
    for route in payload["routes"]:
        assert route["day"]["to"] > route["day"]["from"]
        assert route["day"]["from"] == min(trip["events"][0][1] for trip in route["trips"])
        last = max(trip["events"][-1][1] for trip in route["trips"])
        assert route["day"]["to"] == last


def replay_page_for_test() -> str:
    """Render the replay page on a toy payload, the way `map.write_replay` does."""
    route = {
        "key": "toy|7|ida|weekday",
        "operator": "toy",
        "line": "7",
        "direction": "ida",
        "service": "weekday",
        "label": "7 · ida · weekday",
        "trips_all": 2,
        "boarded": 9,
        "stops": [0, 1, 2],
        "shape": [0, 1, 2],
        "day": {"from": 480, "to": 540},
        "trips": [
            {
                "id": "toy:1",
                "capacity": 40,
                "bus": "std",
                "events": [
                    [0, 480, 3, 0, 3, 3, 0],
                    [1, 490, 2, 1, 4, 2, 0],
                    [2, 500, 0, 4, 0, 0, 0],
                ],
            },
            {
                "id": "toy:2",
                "capacity": 40,
                "bus": "std",
                "events": [
                    [0, 530, 5, 0, 5, 9, 4],
                    [1, 540, 1, 2, 4, 1, 0],
                    [2, 550, 0, 5, 0, 0, 0],
                ],
            },
        ],
    }
    payload = {
        "routes": [route],
        "stops": [
            {
                "id": f"toy:{index}",
                "name": f"Stop {index}",
                "lat": 40.64 + index * 0.01,
                "lon": -8.65,
            }
            for index in range(3)
        ],
        "omitted": 0,
    }
    basemaps = map_module.basemap_payload({"CARTO_API_KEY": ""})
    return pages_module.render_replay(
        payload,
        basemaps=basemaps,
        chosen=next(entry for entry in basemaps if entry["name"] == "network"),
        config=config.ScenarioConfig(),
        report=build.Report(label="toy"),
    )


def test_the_replay_page_offers_the_route_the_clock_and_the_legend() -> None:
    page = replay_page_for_test()
    for control in (
        'id="route"',
        'id="basemap"',
        'id="play"',
        'id="rewind"',
        'id="speed"',
        'id="scrub"',
        'id="clock"',
        'id="stat-live"',
        'id="stat-waiting"',
    ):
        assert control in page, f"the replay lost its {control} control"
    assert "one dot per waiting rider" in page
    assert "left" in page
    assert sorted(set(re.findall(r"__[A-Z_]+__", page))) == []


# -- the supply is real, and that is checked rather than claimed ------------


def test_every_generated_movement_traces_back_to_the_scraped_timetable() -> None:
    """The demand is invented; the supply is not, and a run proves it.

    A future change could quietly synthesise a trip, move a stop or retime a call
    — the tables would still add up. This is the check that would notice.
    """
    network = inputs.read_network()
    root = paths.SCENARIO_DIR / "s0042-random-bpd12-weekday-weekend"
    flows = pl.read_csv(root / "flows.csv", schema_overrides={"line": pl.String})
    od = pl.read_csv(root / "od_flow.csv", schema_overrides={"line": pl.String})
    demand = pl.read_csv(root / "od_demand.csv", schema_overrides={"line": pl.String})
    built = FlowDay(
        trips=flows,
        stops=pl.read_csv(root / "stop_flow.csv", schema_overrides={"line": pl.String}),
        od=od,
    )
    problems, counts = validate.check_provenance(built, demand, network)
    assert problems == []
    assert counts["supply_trips"] == 1429
    assert counts["supply_stops"] >= 1844
    assert counts["generated_trips"] == flows["trip_id"].n_unique()
    assert counts["od_rows"] == od.height


def test_provenance_notices_an_invented_trip_a_moved_clock_and_a_fake_stop(
    toy_timetable: pl.DataFrame,
) -> None:
    """The three ways the output could stop describing the real network."""
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        real = network.served()
        good = FlowDay(
            trips=real.select(
                "trip_id", "stop_id", "stop_seq", "minutes", "line", "direction", "service_group"
            ).with_columns(
                pl.lit("toy").alias("operator"),
                pl.lit(0).cast(pl.Int64).alias("boarded"),
            ),
            stops=pl.DataFrame(),
            od=pl.DataFrame(
                {
                    "trip_id": ["t1"],
                    "origin_id": ["toy:alpha"],
                    "destination_id": ["toy:beta"],
                }
            ),
        )
        problems, _ = validate.check_provenance(good, pl.DataFrame(), network)
        assert problems == []

        invented = good.trips.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit("invented:trip"))
            .otherwise(pl.col("trip_id"))
            .alias("trip_id")
        )
        problems, _ = validate.check_provenance(
            FlowDay(trips=invented, stops=pl.DataFrame(), od=good.od),
            pl.DataFrame(),
            network,
        )
        assert any("not in the scraped timetable" in problem for problem in problems)

        retimed = good.trips.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("minutes") + 7)
            .otherwise(pl.col("minutes"))
            .alias("minutes")
        )
        problems, _ = validate.check_provenance(
            FlowDay(trips=retimed, stops=pl.DataFrame(), od=good.od),
            pl.DataFrame(),
            network,
        )
        assert any("not calls in the timetable" in problem for problem in problems)

        fabricated = good.od.with_columns(pl.lit("toy:nowhere").alias("destination_id"))
        problems, _ = validate.check_provenance(
            FlowDay(trips=good.trips, stops=pl.DataFrame(), od=fabricated),
            pl.DataFrame(),
            network,
        )
        assert any("not in stops.csv" in problem for problem in problems)


def test_a_circular_route_is_not_called_backwards(toy_timetable: pl.DataFrame) -> None:
    """A loop calls at the same stop twice; riding it is not riding backwards."""
    with tempfile.TemporaryDirectory() as directory:
        network = toy_network(Path(directory), toy_timetable)
        loop = pl.DataFrame(
            {
                "trip_id": ["loop"] * 4,
                "operator": ["toy"] * 4,
                "line": ["9"] * 4,
                "direction": ["ida"] * 4,
                "service": ["weekday"] * 4,
                "service_group": ["weekday"] * 4,
                "stop_seq": [1, 2, 3, 4],
                "stop_id": ["toy:alpha", "toy:beta", "toy:alpha", "toy:gamma"],
                "stop": ["Alpha", "Beta", "Alpha", "Gamma"],
                "minutes": [480, 490, 500, 510],
                "served": [True] * 4,
            },
            schema=TOY_COLUMNS,
        )
        supply = inputs.Network(
            stop_times=pl.concat([network.stop_times, loop]), stops=network.stops
        )
        rows = [
            {"trip_id": "loop", "origin_id": origin, "destination_id": destination}
            for origin, destination in (
                ("toy:alpha", "toy:beta"),
                ("toy:beta", "toy:alpha"),
                ("toy:alpha", "toy:gamma"),  # resolves against the *second* Alpha
            )
        ]
        built = FlowDay(
            trips=pl.DataFrame(
                {
                    "trip_id": ["loop"],
                    "stop_id": ["toy:alpha"],
                    "minutes": [480],
                    "line": ["9"],
                    "direction": ["ida"],
                    "service_group": ["weekday"],
                }
            ),
            stops=pl.DataFrame(),
            od=pl.DataFrame(rows),
        )
        problems, _ = validate.check_provenance(built, pl.DataFrame(), supply)
        assert problems == []
