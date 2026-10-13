"""The matrices themselves: aggregated over lines, long or wide.

``wide_matrix`` pivots with ``stop_id`` order, which is alphabetical and says
nothing about the network. A route-ordered pivot belongs to the caller: it needs
a stop order derived from the trips, not from the ids.
"""

from __future__ import annotations

import pandas as pd


def od_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """The OD matrix: direct trips per stop pair, by operator, service and band."""
    if pairs.empty:
        return pairs
    frame = pairs.copy()
    return (
        frame.groupby(
            [
                "operator",
                "service_group",
                "departure_band",
                "origin_id",
                "origin",
                "destination_id",
                "destination",
            ],
            as_index=False,
        )
        .agg(
            trips=("trips", "sum"),
            lines=("line", lambda values: " ".join(sorted(set(values)))),
            first_minutes=("first_minutes", "min"),
            last_minutes=("last_minutes", "max"),
            min_run_min=("min_run_min", "min"),
            median_run_min=("median_run_min", "median"),
            max_run_min=("max_run_min", "max"),
        )
        .sort_values(["operator", "service_group", "departure_band", "origin_id"])
    )


def wide_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long OD matrix into origins by destinations."""
    if matrix.empty:
        return matrix
    return (
        matrix.pivot_table(
            index=["operator", "origin_id", "origin"],
            columns="destination_id",
            values="trips",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .sort_values(["operator", "origin_id"])
    )
