"""
Shared tabular string helpers: postcode normalisation and the England filter.

These two operations were duplicated across the ``data/`` scripts with subtly
different implementations. Centralising them makes the postcode space-handling
an explicit, per-call-site choice and gives one definition of "England only".
"""

from __future__ import annotations

import pandas as pd

#: ONS geography codes for England start with "E" (E00… OAs, E01… LSOAs, etc.).
ENGLAND_CODE_PREFIX = "E"

#: Postcode-area prefixes for Scotland and Wales, excluded from England-only
#: postcode processing. England postcode areas are the complement of this set.
#: Centralised here because it was duplicated verbatim across acquisition scripts.
SCOTTISH_WELSH_POSTCODE_AREAS = frozenset(
    {
        # Scotland
        "AB",
        "DD",
        "DG",
        "EH",
        "FK",
        "G",
        "HS",
        "IV",
        "KA",
        "KW",
        "KY",
        "ML",
        "PA",
        "PH",
        "TD",
        "ZE",
        # Wales
        "CF",
        "LD",
        "LL",
        "NP",
        "SA",
    }
)


def normalise_postcode(s: pd.Series, *, keep_space: bool) -> pd.Series:
    """
    Normalise UK postcodes for matching.

    Always upper-cases and strips surrounding whitespace. The ``keep_space``
    flag selects the space convention, which MUST match the other side of any
    join:

    * ``keep_space=True`` preserves the single internal space of the standard
      "AB10 1AU" form. Used by the metered-energy and EPC paths, which join to
      the Code-Point-derived ``postcode_oa_lookup`` (also spaced).
    * ``keep_space=False`` removes all internal spaces ("AB101AU"). Used by the
      NHS path, which joins ODS postcodes to its own spaceless Code-Point key.

    Both conventions are internally consistent within their own join; mixing
    them across a join is a bug (a spaced key never matches a spaceless one).

    Parameters
    ----------
    s : pd.Series
        Raw postcode strings.
    keep_space : bool
        Keep the internal space (True) or strip all spaces (False).

    Returns
    -------
    pd.Series
        Normalised postcode strings.
    """
    out = s.astype(str).str.strip().str.upper()
    if not keep_space:
        out = out.str.replace(" ", "", regex=False)
    return out


def england_code_mask(codes: pd.Series) -> pd.Series:
    """
    Boolean mask selecting England ONS geography codes (those starting "E").

    Parameters
    ----------
    codes : pd.Series
        ONS geography codes (e.g. OA21CD, LSOA21CD).

    Returns
    -------
    pd.Series
        Boolean mask, True where the code is English.
    """
    return codes.astype(str).str.startswith(ENGLAND_CODE_PREFIX)
