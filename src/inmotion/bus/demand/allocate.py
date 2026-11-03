"""Integer apportionment: splitting a whole number of people without losing one.

The demand across stop pairs, the assignment across the trips serving a pair, and
the queue across the destinations wanting the last seats all use largest
remainder, with ties broken by a seeded jitter so no rounding detail becomes a
systematic bias. ``sampled`` is the fourth: drawing riders individually is what
makes a day sparse with a long tail, where rounding the expected value of every
cell would empty most of the matrix.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import polars as pl


def jitter(seed: int, salt: str, n: int) -> pl.Series:
    """``n`` reproducible uniform draws below 1, for breaking rounding ties.

    The salt is digested with SHA-256 rather than :func:`hash`: CPython salts string
    hashing per process, which would make a "reproducible" scenario differ between
    two runs of the same command.
    """
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    derived = int.from_bytes(digest[:8], "big")
    generator = np.random.default_rng(derived)
    return pl.Series("_jitter", generator.random(n))


def largest_remainder(
    weights: Sequence[float], total: int, *, tie_break: Sequence[float] | None = None
) -> list[int]:
    """Split ``total`` between ``weights``, to integers that add back up to it.

    A group whose weights are all zero has no share to be proportional to, so it
    is split evenly rather than dropped: a row that exists must not vanish from
    the tables just because its weight is zero.
    """
    if total < 0:
        raise ValueError(f"cannot split a negative total: {total}")
    shares = [max(float(weight), 0.0) for weight in weights]
    if not shares:
        return []
    if tie_break is not None and len(tie_break) != len(shares):
        raise ValueError("a tie-break needs one value per weight")
    denominator = sum(shares)
    if denominator <= 0:
        shares = [1.0] * len(shares)
        denominator = float(len(shares))
    exact = [share / denominator * total for share in shares]
    base = [int(value // 1) for value in exact]
    fractions = [value - whole for value, whole in zip(exact, base, strict=True)]
    leftovers = total - sum(base)
    order = sorted(
        range(len(shares)),
        key=lambda index: (
            -fractions[index],
            -(tie_break[index] if tie_break is not None else 0.0),
            index,
        ),
    )
    for index in order[:leftovers]:
        base[index] += 1
    return base


def sampled(
    weights: Sequence[float] | "np.ndarray", total: int, *, seed: int, salt: str
) -> list[int]:
    """Draw ``total`` riders across alternatives, in proportion to ``weights``.

    Right for the day as a whole, where rounding the expected value would give every
    one of 44,000 pairs a third of a rider and empty all but a few hundred: drawing
    each rider's pair makes the matrix sparse with a long tail, as real OD data is.
    The total is exact and the seed replays it; zero weights are never drawn.
    """
    if total < 0:
        raise ValueError(f"cannot draw a negative total: {total}")
    shares = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if shares.size == 0:
        return []
    if shares.sum() <= 0:
        shares = np.ones_like(shares)
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return generator.multinomial(total, shares / shares.sum()).tolist()


def apportion(
    frame: pl.DataFrame,
    *,
    weights: str,
    total: str,
    keys: Sequence[str],
    seed: int,
    salt: str,
    name: str = "riders",
) -> pl.DataFrame:
    """Add ``name``: ``total`` split per ``keys`` group in proportion to ``weights``.

    The jitter only separates equal fractions — a second key, not a perturbation of
    the first, or a lucky draw could take a seat from a measurably larger share.
    """
    # 10**9 separates the two ranks without ever colliding: a group holds at
    # most a few thousand rows, and Int64 holds the product exactly. Both ranks
    # are cast first: polars ranks are unsigned 32-bit, and five times a billion
    # wraps around in that width, which would silently hand the last seat to the
    # smallest share.
    order = (
        (
            pl.col("_fraction_rank").cast(pl.Int64) * 1_000_000_000
            + pl.col("_tie_rank").cast(pl.Int64)
        )
        .rank(method="ordinal")
        .over(keys)
    )
    share = (
        pl.col(total).cast(pl.Float64)
        * pl.col(weights).clip(lower_bound=0)
        / pl.col(weights).clip(lower_bound=0).sum().over(keys)
    )
    return (
        frame.with_columns(jitter(seed, salt, frame.height))
        .with_columns(share.alias("_share"))
        .with_columns(
            pl.col("_share").floor().alias("_base"),
            (pl.col("_share") - pl.col("_share").floor()).alias("_frac"),
        )
        .with_columns(
            pl.col("_frac")
            .rank(method="ordinal", descending=True)
            .over(keys)
            .alias("_fraction_rank"),
            pl.col("_jitter").rank(method="ordinal", descending=True).over(keys).alias("_tie_rank"),
            (pl.col(total).cast(pl.Int64) - pl.col("_base").sum().over(keys)).alias("_left"),
        )
        .with_columns(order.alias("_order"))
        .with_columns(
            (pl.col("_base") + (pl.col("_order") <= pl.col("_left")).cast(pl.Int64))
            .cast(pl.Int64)
            .alias(name)
        )
        .drop(
            "_share", "_base", "_frac", "_fraction_rank", "_tie_rank", "_left", "_order", "_jitter"
        )
    )
