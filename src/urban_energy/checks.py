"""
Join-safety helpers: match-rate floors that raise on excessive row loss.

An inner join that silently drops a large fraction of the left frame usually
signals a key-normalisation or coverage regression (for example, postcodes that
stopped matching the lookup after an upstream format change). These helpers make
such a drop fail loudly instead of shrinking the analysis sample unnoticed.
"""

from __future__ import annotations

#: Maximum fraction of the left frame an inner join may drop before it is
#: treated as a data error. Five percent tolerates the normal tail of
#: unmatchable rows (retired postcodes, boundary edge cases) while catching a
#: gross regression where a whole key column stops matching.
DEFAULT_MAX_DROP_FRACTION = 0.05


def assert_match_rate(
    n_left: int,
    n_matched: int,
    *,
    name: str,
    max_drop_fraction: float = DEFAULT_MAX_DROP_FRACTION,
) -> None:
    """
    Log the row loss from an inner join and raise if it exceeds the floor.

    Parameters
    ----------
    n_left : int
        Number of rows in the left frame before the join.
    n_matched : int
        Number of rows retained after the inner join.
    name : str
        Human-readable label for the join (used in the log line and error).
    max_drop_fraction : float
        Maximum tolerated fraction of dropped rows before raising.

    Raises
    ------
    ValueError
        If the left frame is empty, or if the dropped fraction exceeds
        ``max_drop_fraction``.
    """
    if n_left <= 0:
        raise ValueError(f"{name}: left frame is empty (n_left={n_left}).")

    dropped = n_left - n_matched
    frac = dropped / n_left
    print(
        f"  {name}: matched {n_matched:,}/{n_left:,} (dropped {dropped:,}, {frac:.1%})"
    )
    if frac > max_drop_fraction:
        raise ValueError(
            f"{name}: inner join dropped {dropped:,}/{n_left:,} rows "
            f"({frac:.1%}), exceeding the {max_drop_fraction:.0%} floor. "
            "This usually means a key-normalisation or coverage regression; "
            "inspect the join keys before trusting the output."
        )
