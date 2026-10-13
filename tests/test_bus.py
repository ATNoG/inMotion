"""Bus-network ingestion: the BusWay table parser, the OD maths and the PDF grid.

The AveiroBus parser reads pixels, so its OCR is exercised against a cached PDF
when one is present and skipped otherwise — and skipped too when tesseract has
no Portuguese language pack, since that is the model the readings came from.
Everything around the OCR (the geometry, the time format, the stop keys, the
reconciliation rules) is tested directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from inmotion.bus import aveirobus, busway, od, viz
from inmotion.bus.records import Call, Trip, roll_past_midnight

BUSWAY_PAGE = """
<table class='tabela-horaria'><tbody>
<tr><td class='numero_linha_1_20' colspan='2'><b>16</b> &#8211; IDA <i>OUTBOUND</i></td>
<td class='funcionamento' colspan='3'><b>Dias úteis</b><br><i>Working Days</i></td></tr>
<tr class='epoca_1_20'><td class='epoca_titulo_1_20' colspan='2'><b>ÉPOCA</b></td>
<td class='epoca_1_20'>A</td><td class='epoca_1_20'>PE</td><td class='epoca_1_20'>A</td></tr>
<tr class='claro'><td class='seta'>↓</td><td class='paragem'>Estação (Aveiro)</td>
<td>06:40</td><td>07:30</td><td>08:20</td></tr>
<tr class='escuro'><td class='seta'>↓</td><td class='paragem'>Areal (Angeja)</td>
<td>07:09</td><td class='fundoamarelo'>07:58*</td>
<td>–</td></tr>
<tr><td class='observacao' colspan='2'>*</td>
<td class='observacoes' colspan='3'>Ligação à linha 33</td></tr>
</tbody></table>
"""

SECOND_TEMPLATE = """
<table class='tabela-horaria'><tbody>
<tr><td class='numero_linha_M_Sever_Vouga' colspan='2'><b>100</b> – IDA</td>
<td class='funcionamento' colspan='2'>Sábados</td></tr>
<tr><td class='epoca_titulo_M_Sever_Vouga' colspan='2'>ÉPOCA</td>
<td class='epoca_M_Sever_Vouga'>A</td><td class='epoca_M_Sever_Vouga'>A</td></tr>
<tr><td class='seta'>↓</td><td class='paragem'>Macida</td><td>07:30</td><td>–</td></tr>
</tbody></table>
"""


def test_busway_page_parses_every_trip_and_call() -> None:
    trips = busway.parse_line_page(
        BUSWAY_PAGE, operator="busway", line="16", direction="ida", source="test", page_key="16-ida"
    )
    assert len(trips) == 3
    assert [trip.season for trip in trips] == ["A", "PE", "A"]
    assert {trip.service for trip in trips} == {"weekday"}
    assert all(trip.line == "16" and trip.direction == "ida" for trip in trips)

    first = trips[0]
    assert [call.stop for call in first.calls] == ["Estação (Aveiro)", "Areal (Angeja)"]
    assert [call.time for call in first.calls] == ["06:40", "07:09"]

    # A dash is "this trip does not call here", not a missing reading.
    skipped = trips[2].calls[1]
    assert skipped.time == ""
    assert skipped.minutes is None

    # The starred, highlighted cell carries both the footnote and the highlight.
    marked = trips[1].calls[1]
    assert marked.time == "07:58"
    assert "*" in marked.markers
    assert "!" in marked.markers

    # No trip ever runs backwards.
    for trip in trips:
        minutes = [call.minutes for call in trip.calls if call.minutes is not None]
        assert minutes == sorted(minutes)


def test_busway_tables_do_not_share_their_trips() -> None:
    """A page carries one table per service, each numbering its trips from 1.

    Indexing the page-wide trip list made the second table append its own
    departures onto the first table's trips, which doubled every call list and
    interleaved two services into one trip.
    """
    weekend = BUSWAY_PAGE.replace("Dias úteis", "Fins de Semana e Feriados").replace(
        "<br><i>Working Days</i>", ""
    )
    trips = busway.parse_line_page(
        BUSWAY_PAGE + weekend,
        operator="busway",
        line="16",
        direction="ida",
        source="test",
        page_key="16-ida",
    )
    assert len(trips) == 6
    assert {trip.service for trip in trips} == {"weekday", "weekend"}
    for trip in trips:
        assert len(trip.calls) == 2
        assert [call.stop for call in trip.calls] == ["Estação (Aveiro)", "Areal (Angeja)"]


def test_busway_reads_the_second_table_template() -> None:
    trips = busway.parse_line_page(
        SECOND_TEMPLATE,
        operator="busway",
        line="100",
        direction="ida",
        source="test",
        page_key="100-ida",
    )
    assert len(trips) == 2
    assert all(trip.service == "saturday" for trip in trips)
    assert trips[0].calls[0].minutes == 7 * 60 + 30
    assert trips[1].calls[0].minutes is None


def test_busway_slug_wins_over_a_stale_heading() -> None:
    """The 15-volta page's heading says IDA, because the site copy-pasted it."""
    stale = BUSWAY_PAGE.replace("IDA <i>OUTBOUND</i>", "IDA <i>OUTBOUND</i>")
    line, direction, caption = busway.page_identity(stale, line="15", direction="volta")
    assert (line, direction) == ("15", "volta")

    # A slug that names no direction still gets one from the heading.
    line, direction, _ = busway.page_identity(stale, line="35", direction="")
    assert (line, direction) == ("35", "ida")
    assert "IDA" in caption


def test_trip_ids_are_unique_per_page() -> None:
    weekend = BUSWAY_PAGE.replace("Dias úteis", "Fins de Semana e Feriados")
    first = busway.parse_line_page(
        BUSWAY_PAGE,
        operator="busway",
        line="15",
        direction="ida",
        source="a",
        page_key="15-ida",
    )
    second = busway.parse_line_page(
        weekend,
        operator="busway",
        line="15",
        direction="ida",
        source="b",
        page_key="15-volta",
    )
    assert not {trip.trip_id for trip in first} & {trip.trip_id for trip in second}


def test_minutes_past_midnight_are_kept() -> None:
    trips = busway.parse_line_page(
        BUSWAY_PAGE.replace("06:40", "25:10"),
        operator="busway",
        line="16",
        direction="ida",
        source="t",
        page_key="16-ida",
    )
    assert trips[0].calls[0].minutes == 25 * 60 + 10


def test_stop_keys_survive_ocr_artefacts() -> None:
    assert od.canonical("Qt? Clementina") == od.canonical("Qtª Clementina")
    assert od.canonical("Hospital / Universidade A") == od.canonical("Hospital Universidade A")
    assert od.stop_id("aveirobus", "Qt? Clementina") == "aveirobus:qt-clementina"


def test_departure_bands() -> None:
    assert od.band(6 * 60) == "early"
    assert od.band(8 * 60) == "am_peak"
    assert od.band(12 * 60) == "midday"
    assert od.band(18 * 60) == "pm_peak"
    assert od.band(22 * 60) == "evening"
    assert od.band(25 * 60) == "night"


def _trip(trip_id: str, times: list[tuple[str, int | None]], service: str = "weekday") -> Trip:
    return Trip(
        trip_id=trip_id,
        operator="busway",
        line="16",
        direction="ida",
        service=service,
        season="A",
        trip_index=1,
        source="test",
        calls=[
            Call(stop_seq=i + 1, stop=stop, time=time, minutes=minutes)
            for i, (stop, (time, minutes)) in enumerate(
                zip(["A", "B", "C", "D"], times, strict=False)
            )
        ],
    )


def _route_trip(trip_id: str, calls: list[tuple[str, int]], service: str = "weekday") -> Trip:
    """A trip with explicit stop names, so route order can differ from alphabetical order.

    ``_trip`` always names its stops A, B, C, D in order, which would make an
    ordering test pass for the wrong reason.
    """
    return Trip(
        trip_id=trip_id,
        operator="busway",
        line="16",
        direction="ida",
        service=service,
        season="A",
        trip_index=1,
        source="test",
        calls=[
            Call(
                stop_seq=i + 1,
                stop=stop,
                time=f"{minutes // 60:02d}:{minutes % 60:02d}",
                minutes=minutes,
            )
            for i, (stop, minutes) in enumerate(calls)
        ],
    )


def test_two_trip_od_matrix_is_triangular_only_in_route_order() -> None:
    """Two trips running Z -> A -> M.

    Alphabetical stop order (A, M, Z) is *not* triangular; route order
    (Z, A, M) is strictly upper triangular. Triangularity is a property of the
    ordering, not of the data, so a matrix shipped in stop_id order can never
    show it.
    """
    trips = [
        _route_trip("t1", [("Z", 360), ("A", 370), ("M", 380)]),
        _route_trip("t2", [("Z", 420), ("A", 430), ("M", 440)]),
    ]
    matrix = od.od_matrix(od.od_pairs(od.stop_times_frame(trips)))

    # 3 pairs per band, and the 06:00 / 07:00 departures fall in different bands.
    assert len(matrix) == 6
    assert set(matrix["departure_band"]) == {"early", "am_peak"}
    trips_per_pair = matrix.groupby(["origin_id", "destination_id"])["trips"].sum()
    assert set(trips_per_pair) == {2}
    assert set(trips_per_pair.index) == {
        ("busway:z", "busway:a"),
        ("busway:z", "busway:m"),
        ("busway:a", "busway:m"),
    }

    def halves(order: list[str]) -> tuple[float, float]:
        idx = {stop: i for i, stop in enumerate(order)}
        m = np.zeros((len(order), len(order)))
        for (origin, destination), trips_ in trips_per_pair.items():
            m[idx[origin], idx[destination]] = trips_
        return float(np.triu(m, 1).sum()), float(np.tril(m, -1).sum())

    # stop_id order is alphabetical: only Z->A and Z->M land above the diagonal.
    assert halves(["busway:a", "busway:m", "busway:z"]) == (2.0, 4.0)
    # Route order puts every ride above the diagonal, and nothing on it.
    assert halves(["busway:z", "busway:a", "busway:m"]) == (6.0, 0.0)


def test_od_pairs_cover_every_ordered_pair_that_is_served() -> None:
    trips = [
        _trip("t1", [("06:00", 360), ("06:10", 370), ("06:20", 380), ("06:30", 390)]),
        _trip("t2", [("07:00", 420), ("07:12", 432), ("", None), ("07:40", 460)]),
    ]
    stop_times = od.stop_times_frame(trips)
    pairs = od.od_pairs(stop_times)

    assert set(pairs["origin"]) <= {"A", "B", "C", "D"}

    # t2 never calls at C, so only t1 connects A to C.
    ac = pairs[(pairs["origin"] == "A") & (pairs["destination"] == "C")].iloc[0]
    assert ac["trips"] == 1
    assert ac["min_run_min"] == ac["max_run_min"] == 20

    # Skipping C does not break A→D on t2, it just takes longer. A pair is one row
    # per departure band now, so A→D splits on the 06:00 (early) and 07:00 (am_peak)
    # departures instead of collapsing into a single whole-day row.
    ad = pairs[(pairs["origin"] == "A") & (pairs["destination"] == "D")]
    assert ad["trips"].sum() == 2
    assert sorted(ad["departure_band"]) == ["am_peak", "early"]
    assert (ad["min_run_min"].min(), ad["median_run_min"].median(), ad["max_run_min"].max()) == (
        30,
        35,
        40,
    )
    assert (ad["first_departure"].min(), ad["last_departure"].max()) == ("06:00", "07:00")

    # Nothing is ever reported in the reverse direction.
    assert not ((pairs["origin"] == "D") & (pairs["destination"] == "A")).any()

    # Only t1 calls at C, so every pair touching C is a single-trip connection.
    touching_c = pairs[(pairs["origin"] == "C") | (pairs["destination"] == "C")]
    assert len(touching_c) == 3
    assert set(touching_c["trips"]) == {1}


def test_od_matrix_aggregates_by_service_and_band() -> None:
    weekday = _trip("t1", [("06:00", 360), ("06:10", 370), ("06:20", 380), ("06:30", 390)])
    saturday = _trip(
        "t2", [("06:00", 360), ("06:10", 370), ("06:20", 380), ("06:30", 390)], "saturday"
    )
    matrix = od.od_matrix(od.od_pairs(od.stop_times_frame([weekday, saturday])))

    cell = matrix[
        (matrix["service_group"] == "weekday")
        & (matrix["departure_band"] == "early")
        & (matrix["origin"] == "A")
        & (matrix["destination"] == "B")
    ].iloc[0]
    assert cell["trips"] == 1
    assert set(matrix["service_group"]) == {"weekday", "weekend"}

    wide = od.wide_matrix(
        matrix[matrix["service_group"] == "weekday"].drop(
            columns=["service_group", "departure_band"]
        )
    )
    assert wide.loc[wide["origin"] == "A", "busway:b"].iloc[0] == 1


def test_aveirobus_geometry_and_time_helpers() -> None:
    assert aveirobus.runs(np.array([False, True, True, False, True, True, True]), 2) == [
        (1, 3),
        (4, 7),
    ]
    closed = aveirobus.close_horizontal(
        np.array([[True, False, False, True, False, False, False]]), 2
    )
    assert closed[0, :6].all()  # both blocks of red bridge their two-pixel gaps
    assert not closed[0, 6]
    assert aveirobus.to_minutes("25:05") == 25 * 60 + 5
    assert aveirobus.to_minutes("–") is None
    # Clustering chains: a value joins the previous group while it is within
    # tolerance of that group's last member, which is what keeps a row's
    # slightly-baseline-shifted times together.
    assert aveirobus.cluster([1.0, 1.5, 5.0], 0.6) == [(1.25, 2), (5.0, 1)]


def test_times_past_midnight_take_a_consistent_scale() -> None:
    trip = Trip(
        trip_id="t",
        operator="busway",
        line="1",
        direction="ida",
        service="weekday",
        season="A",
        trip_index=1,
        source="test",
        calls=[
            Call(stop_seq=1, stop="A", time="23:58", minutes=23 * 60 + 58),
            Call(stop_seq=2, stop="B", time="00:02", minutes=2),
            Call(stop_seq=3, stop="C", time="00:09", minutes=9),
        ],
    )
    # One crossing, but every later call moves with it.
    assert roll_past_midnight(trip) == 1
    assert [call.time for call in trip.calls] == ["23:58", "24:02", "24:09"]
    assert [call.minutes for call in trip.calls] == [1438, 1442, 1449]

    # A trip that does not cross midnight is left exactly as printed.
    day = Trip(
        trip_id="d",
        operator="busway",
        line="1",
        direction="ida",
        service="weekday",
        season="A",
        trip_index=1,
        source="test",
        calls=[
            Call(stop_seq=1, stop="A", time="08:00", minutes=480),
            Call(stop_seq=2, stop="B", time="08:10", minutes=490),
        ],
    )
    assert roll_past_midnight(day) == 0
    assert [call.time for call in day.calls] == ["08:00", "08:10"]


def _block(times: list[list[str]]) -> aveirobus.Block:
    return aveirobus.Block(
        index=0,
        top=0,
        bottom=100,
        columns=[10.0, 60.0],
        rows=[20.0, 60.0],
        stops=["A", "B"],
        times=times,
        colours=["", ""],
    )


def test_reconcile_never_blanks_a_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    # The crop reads nothing at all, which is what happens on a blank cell.
    blank = Image.new("RGB", (200, 100), "white")
    # The reconciliation rules are the subject here, not the OCR engine, so the
    # third reading is stubbed rather than run: the bitmap test below is the one
    # that goes through tesseract.
    monkeypatch.setattr(aveirobus, "read_cell", lambda image, box: "")

    # Both passes read times and disagree: the first stands and the clash is
    # recorded, because emptying the cell would lose a real departure.
    block = _block([["06:30", "07:00"], ["06:40", "07:10"]])
    aveirobus.reconcile(blank, block, [["06:30", "08:00"], ["06:40", ""]])
    assert block.times[0][1] == "07:00"
    assert block.times[1][1] == "07:10"
    assert any("disagreed" in warning for warning in block.warnings)

    # A time only the second pass saw is filled in, not thrown away.
    seen_by_second = _block([["", ""], ["06:40", "07:10"]])
    aveirobus.reconcile(blank, seen_by_second, [["", "07:10"], ["06:40", "07:10"]])
    assert seen_by_second.times[0][1] == "07:10"
    assert seen_by_second.times[0][0] == ""


def test_tesseract_without_tsv_is_an_error_not_an_empty_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tesseract that cannot render TSV must fail, not read every cell blank.

    It exits 0 and falls back to plain text when the ``tsv`` config is missing
    from the tessdata directory, which parses into no words at all: captions come
    back empty, no header band is found, and a whole build is skipped as
    unreadable without a single error.
    """
    plain = SimpleNamespace(
        returncode=0,
        stdout=b"06:43 07:30\n",
        stderr=b"read_params_file: Can't open tsv",
    )
    monkeypatch.setattr(aveirobus, "subprocess", SimpleNamespace(run=lambda *a, **k: plain))

    with pytest.raises(RuntimeError, match="produced no TSV"):
        aveirobus.read_cell(Image.new("RGB", (60, 30), "white"), (0, 0, 60, 30))


def test_monotonic_pass_keeps_the_reading_it_cannot_improve() -> None:
    block = aveirobus.Block(
        index=0,
        top=0,
        bottom=100,
        columns=[10.0, 60.0],
        rows=[20.0, 60.0],
        stops=["A", "B"],
        times=[["06:50", ""], ["06:30", ""]],
        colours=["", ""],
    )
    aveirobus.enforce_monotonic(None, block)
    assert block.times[1][0] == "06:30"
    assert any("06:30" in warning for warning in block.warnings)


def _route_frame(route: str, line: str, direction: str, stops: list[tuple[str, int]]) -> Trip:
    """A trip whose stops and times are given as (name, minutes) in order."""
    service = "weekday"
    return Trip(
        trip_id=f"{route}:{line}:{direction}:1",
        operator=route,
        line=line,
        direction=direction,
        service=service,
        season="A",
        trip_index=1,
        source="test",
        calls=[
            Call(
                stop_seq=i + 1,
                stop=stop,
                time=f"{minutes // 60:02d}:{minutes % 60:02d}",
                minutes=minutes,
            )
            for i, (stop, minutes) in enumerate(stops)
        ],
    )


def test_similar_pairs_ranks_a_reversal_above_an_unrelated_route() -> None:
    """A line's two directions share every stop in reverse order; a stranger does not."""
    out = _route_frame("op", "1", "ida", [("A", 0), ("B", 5), ("C", 12), ("D", 20)])
    back = _route_frame("op", "1", "volta", [("D", 0), ("C", 7), ("B", 14), ("A", 21)])
    other = _route_frame("op", "2", "ida", [("A", 0), ("B", 4), ("X", 9), ("Y", 15)])
    stop_times = od.stop_times_frame([out, back, other])

    ranked = viz.similar_pairs(stop_times, "op", minimum_stops=3)
    assert ranked, "a pair of comparable routes should have been found"
    best = ranked[0]
    assert {best.first.line, best.second.line} == {"1"}
    # The two directions call at the same stops in opposite order.
    assert best.shared == 4
    assert best.jaccard == 1.0
    assert best.agreement == pytest.approx(-1.0)

    # The stranger shares half the stops, in the same order, and ranks below.
    matching = [p for p in ranked if {p.first.line, p.second.line} == {"1", "2"}]
    assert matching
    assert matching[0].jaccard < best.jaccard
    assert matching[0].agreement > 0


def test_similar_pairs_ignores_routes_that_are_too_short() -> None:
    trips = [
        _route_frame("op", "1", "ida", [("A", 0), ("B", 5)]),
        _route_frame("op", "1", "volta", [("B", 0), ("A", 5)]),
    ]
    assert viz.similar_pairs(od.stop_times_frame(trips), "op", minimum_stops=3) == []


def test_od_grid_is_upper_triangular() -> None:
    """A route never serves a pair backwards and never a stop to itself."""
    trip = _route_frame("op", "1", "ida", [("A", 0), ("B", 5), ("C", 12)])
    stop_times = od.stop_times_frame([trip])
    pairs = od.od_pairs(stop_times)
    route = viz.Route("op", "1", "ida", "weekday")
    profile = viz.route_profile(stop_times, route)
    grid = viz.od_grid(pairs, route, list(profile["stop_id"]))

    values = grid.to_numpy()
    # Every stop but the last is an origin; the terminal one has nowhere to go.
    assert (values[:-1].sum(axis=1) > 0).all()
    assert (values[-1] == 0).all()
    assert (values[np.tril_indices_from(values, k=0)] == 0).all()
    assert (values[0, 1], values[0, 2], values[1, 2]) == (1, 1, 1)


CACHED_PDF = Path("data/bus/raw/aveirobus/aveiro-bus-site-horario-linha-297x297-l1-202506.pdf")


@pytest.mark.skipif(not CACHED_PDF.exists(), reason="AveiroBus PDF cache is absent")
@pytest.mark.skipif(
    aveirobus.OCR_LANGUAGE not in aveirobus.tesseract_languages(),
    reason=f"tesseract has no '{aveirobus.OCR_LANGUAGE}' language pack",
)
def test_aveirobus_reads_line_one_from_the_bitmap() -> None:
    extraction = aveirobus.parse_pdf(CACHED_PDF, line="1")
    assert extraction.caption.startswith("CACIA")
    assert extraction.trips, "no trips recovered"

    blocks = {trip.block for trip in extraction.trips}
    assert blocks == {1, 2}
    for block in blocks:
        trips = [trip for trip in extraction.trips if trip.block == block]
        assert len(trips[0].calls) == 14
        assert {trip.service for trip in trips} == {"weekday", "weekend"}

    block_one = [trip for trip in extraction.trips if trip.block == 1]
    first = next(trip for trip in block_one if trip.trip_index == 1)
    assert [call.time for call in first.calls][:3] == ["06:43", "06:46", "06:54"]
    assert first.calls[-1].time == "07:29"

    # Only the third column is tinted, and the tint is the operator's way of
    # saying the trip runs in term time only.
    assert [trip.trip_index for trip in block_one if trip.school_only] == [3]
    assert {trip.colour for trip in block_one if trip.school_only} == {"#a890c0"}
