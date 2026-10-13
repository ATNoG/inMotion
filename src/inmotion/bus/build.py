"""Build the bus tables and the OD matrices from the two operator sites.

One pass, three stages: fetch (cached, so it only hits the network once), parse
each operator with its own parser, then merge into the tidy tables and the OD
matrices. The two parsers are independent, so a failure on one network still
leaves the other's tables written and reports what went wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from inmotion.bus import aveirobus, busway, od, paths, sources
from inmotion.bus.records import Trip


@dataclass
class Report:
    """What a build produced, and what it had to work around."""

    busway_pages: int = 0
    aveirobus_pdfs: int = 0
    trips: int = 0
    stop_times: int = 0
    od_pairs: int = 0
    od_cells: int = 0
    ocr_repairs: int = 0
    crossings: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"busway pages   {self.busway_pages}",
            f"aveirobus pdfs {self.aveirobus_pdfs}",
            f"trips          {self.trips}",
            f"stop times     {self.stop_times}",
            f"od pairs       {self.od_pairs}",
            f"od cells       {self.od_cells}",
            f"ocr repairs    {self.ocr_repairs}",
            f"midnight rolls {self.crossings}",
            f"warnings       {len(self.warnings)}",
        ]
        return "\n".join(lines)


def parse_busway(report: Report, *, fetcher: sources.Fetcher) -> list[Trip]:
    """Every BusWay trip, one line page at a time."""
    trips: list[Trip] = []
    for page in sources.busway_inventory(fetcher):
        report.busway_pages += 1
        html = page.path.read_text(errors="replace")
        found = busway.parse_line_page(
            html,
            operator=page.operator,
            line=page.line,
            direction=page.direction,
            source=page.url,
            page_key=page.url.rstrip("/").rsplit("/", 1)[-1],
        )
        if not found:
            report.warnings.append(f"busway {page.line}-{page.direction}: no timetable tables")
        trips.extend(found)
    return trips


def parse_aveirobus(report: Report, *, fetcher: sources.Fetcher) -> list[Trip]:
    """Every AveiroBus trip, one PDF at a time."""
    # Refuse before writing anything: a build without the language pack would
    # produce BusWay-only tables that look like the whole network.
    if aveirobus.OCR_LANGUAGE not in aveirobus.tesseract_languages():
        raise RuntimeError(
            f"AveiroBus timetables are bitmaps, so reading them needs tesseract with the "
            f"'{aveirobus.OCR_LANGUAGE}' language pack "
            f"(Debian/Ubuntu: {aveirobus.OCR_LANGUAGE_PACKAGE}). Nothing was written."
        )
    trips: list[Trip] = []
    for page in sources.aveirobus_inventory(fetcher):
        report.aveirobus_pdfs += 1
        try:
            extraction = aveirobus.parse_pdf(page.path, operator=page.operator, line=page.line)
        except (ValueError, OSError) as error:
            report.warnings.append(f"aveirobus {page.line}: {error}")
            continue
        report.ocr_repairs += extraction.repaired
        report.crossings += extraction.crossings
        report.warnings.extend(f"aveirobus {page.line}: {w}" for w in extraction.warnings)
        trips.extend(extraction.trips)
    return trips


def collect(*, refresh: bool = False, verbose: bool = True) -> tuple[list[Trip], Report]:
    """Fetch both operators and parse everything into trips."""
    report = Report()
    trips: list[Trip] = []
    with sources.Fetcher(force=refresh, verbose=verbose) as fetcher:
        trips.extend(parse_busway(report, fetcher=fetcher))
        trips.extend(parse_aveirobus(report, fetcher=fetcher))
    report.trips = len(trips)
    return trips, report


def write_tables(trips: list[Trip], report: Report) -> dict[str, pd.DataFrame]:
    """Write the tidy tables and the OD matrices, and return what was written."""
    stop_times = od.stop_times_frame(trips)
    trips_table = od.trips_frame(trips)
    stops = od.stops_frame(stop_times)
    pairs = od.od_pairs(stop_times)
    matrix = od.od_matrix(pairs)

    report.warnings.extend(od.validate(stop_times, pairs))
    report.stop_times = len(stop_times)
    report.od_pairs = len(pairs)
    report.od_cells = len(matrix)

    # Interim copies are per operator, so a parser can be checked on its own.
    for operator in sorted(stop_times["operator"].unique()):
        stop_times[stop_times["operator"] == operator].to_csv(
            paths.interim(f"{operator}_stop_times.csv"), index=False
        )
    trips_table.to_csv(paths.processed("trips.csv"), index=False)
    stops.to_csv(paths.processed("stops.csv"), index=False)
    stop_times.to_csv(paths.processed("stop_times.csv"), index=False)
    pairs.to_csv(paths.processed("od_pairs.csv"), index=False)
    matrix.to_csv(paths.processed("od_matrix.csv"), index=False)
    # One matrix per operator: the two networks share no stop ids, so a single
    # grid over their union is two disjoint blocks in 1,800-odd columns. The
    # long form is written per operator too, since that is what a study loads.
    for operator in sorted(matrix["operator"].unique()):
        mine = matrix[matrix["operator"] == operator]
        mine.drop(columns=["operator"]).to_csv(
            paths.processed(f"od_matrix_{operator}.csv"), index=False
        )
        for group in ("weekday", "weekend"):
            subset = mine[mine["service_group"] == group].drop(
                columns=["service_group", "departure_band"]
            )
            # wide_matrix keys its rows by operator, so it is dropped after the
            # pivot rather than before it.
            od.wide_matrix(subset).drop(columns=["operator"]).to_csv(
                paths.processed(f"od_matrix_wide_{operator}_{group}.csv"), index=False
            )

    return {
        "stop_times": stop_times,
        "trips": trips_table,
        "stops": stops,
        "od_pairs": pairs,
        "od_matrix": matrix,
    }


def write_summary(report: Report) -> None:
    """Record the build's counts and warnings next to the tables."""
    payload = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "busway_pages": report.busway_pages,
        "aveirobus_pdfs": report.aveirobus_pdfs,
        "trips": report.trips,
        "stop_times": report.stop_times,
        "od_pairs": report.od_pairs,
        "od_cells": report.od_cells,
        "ocr_repairs": report.ocr_repairs,
        "midnight_rolls": report.crossings,
        "warnings": report.warnings,
    }
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (paths.PROCESSED_DIR / "build.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def run(*, refresh: bool = False, verbose: bool = True) -> int:
    """Collect, write and summarise. Returns a process exit code."""
    trips, report = collect(refresh=refresh, verbose=verbose)
    if not trips:
        print("no trips parsed; nothing written")
        return 1
    write_tables(trips, report)
    write_summary(report)
    print(report.summary())
    for warning in report.warnings[:20]:
        print(f"  ! {warning}")
    if len(report.warnings) > 20:
        print(f"  ! ... and {len(report.warnings) - 20} more in build.json")
    return 0
