"""Invariants the tables have to hold before an OD matrix is worth anything.

Only checks that a wrong answer cannot hide behind: a time outside the service
day, or a journey that arrives before it departs.
"""

from __future__ import annotations

import pandas as pd


def validate(stop_times: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """Invariants the tables have to satisfy to be worth an OD matrix.

    Only checks that a wrong answer cannot hide behind: a time outside the
    service day, or a journey that arrives before it departs.
    """
    problems: list[str] = []
    served = stop_times[stop_times["served"]]
    outside = served[(served["minutes"] < 0) | (served["minutes"] >= 48 * 60)]
    if len(outside):
        problems.append(f"{len(outside)} stop times outside 00:00-48:00")

    backwards = pairs[pairs["min_run_min"] < 0]
    if len(backwards):
        examples = ", ".join(
            f"{row.origin}->{row.destination}" for row in backwards.head(3).itertuples(index=False)
        )
        problems.append(f"{len(backwards)} od pairs run backwards ({examples})")

    ordered = served.sort_values(["trip_id", "stop_seq"])
    backwards_trips = int(
        (ordered.groupby("trip_id")["minutes"].diff() < 0).groupby(ordered["trip_id"]).any().sum()
    )
    if backwards_trips:
        problems.append(f"{backwards_trips} trips have times that run backwards")

    empty = pairs[pairs["trips"] <= 0]
    if len(empty):
        problems.append(f"{len(empty)} od pairs with no trips")
    return problems
