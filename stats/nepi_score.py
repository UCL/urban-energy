"""
NEPI score — per-OA A–G label on the access-per-energy rate.

Implements ``dissemination/score_spec.md`` (Phase 1). The score is descriptive
of the place as-lived; nothing here is modelled. Per OA:

* **rate** — amenities reachable within the OA's own car catchment
  (``net_amen``, from the network cache) divided by its total energy
  (metered home energy plus NTS-anchored car-travel energy,
  kWh/household/yr). The paper's headline rate divides by car energy only;
  the certificate divides by total energy so that every technology lever
  moves the grade (decision 2026-07-29, score_spec.md).
* **letters** — A–G bands cut as household-weighted septiles of the 2021
  national distribution: the rate (best high), total energy per dwelling
  (best low) and the on-foot amenity count (best high). Thresholds are
  computed once, written to ``dissemination/nepi_bands_2021.json``, and then
  **frozen**: reruns load the file, so re-scores show real movement.
* **potential** — the same letters under full technology deployment
  (fabric + heat pump + EV at 100%, the ``scenarios.py`` transforms).
  Access is unchanged by construction, so only the energy side moves.

OAs with missing inputs are flagged, not dropped (``flag_no_epc`` falls back
to a fabric factor of 1, ``flag_low_meters`` marks under-metered areas); only
OAs absent from the network cache cannot be scored at all.

Output: ``$DATA_DIR/statistics/oa_nepi_score.parquet``.

Run:
    uv run python stats/nepi_score.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from oa_data import load_and_aggregate
from scenarios import BOILER_EFF, COP, _fabric_factor
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile

from urban_energy.paths import DATA_DIR, PROJECT_DIR

BANDS_PATH = PROJECT_DIR / "dissemination" / "nepi_bands_2021.json"
NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"
OUT_PATH = DATA_DIR / "statistics" / "oa_nepi_score.parquet"

LETTERS = "ABCDEFG"
N_BANDS = 7
# v2: the rate denominator changed from car-travel energy to total energy
# (home + travel), so the bands were re-frozen on the new distribution.
BANDS_VERSION = "NEPI-2021 v2"

# Elec-meter-to-household ratio below which the metered energy is treated as
# under-recorded (same denominator check as form_size_decomposition §7).
LOW_METER_RATIO = 0.8


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def weighted_thresholds(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    n_bands: int = N_BANDS,
) -> list[float]:
    """Interior weighted-quantile cuts splitting ``values`` into equal-weight bands.

    Quantiles commute with monotone transforms, so weighted septiles of the raw
    rate equal the exponentiated septiles of its log (score_spec.md §Bands).

    Parameters
    ----------
    values : array-like
        The quantity to band (rate, energy, access count).
    weights : array-like
        Weights (households); rows with non-finite value or weight are dropped.
    n_bands : int, default 7
        Number of bands; returns ``n_bands - 1`` ascending thresholds.

    Returns
    -------
    list of float
        Ascending interior thresholds.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    q = (np.cumsum(w) - 0.5 * w) / w.sum()
    fracs = np.arange(1, n_bands) / n_bands
    return [float(x) for x in np.interp(fracs, q, v)]


def assign_letters(
    values: pd.Series | np.ndarray,
    thresholds: list[float],
    best: str,
) -> pd.Series:
    """Assign A–G letters against frozen ascending ``thresholds``.

    Parameters
    ----------
    values : array-like
        The quantity to letter.
    thresholds : list of float
        Ascending interior thresholds (``N_BANDS - 1`` of them).
    best : {'high', 'low'}
        Whether high values (rate, access) or low values (energy) earn an A.

    Returns
    -------
    pandas.Series
        String letters, ``pd.NA`` where the value is missing.
    """
    if best not in ("high", "low"):
        raise ValueError(f"best must be 'high' or 'low', got {best!r}")
    v = _num(pd.Series(values))
    t = np.asarray(thresholds, dtype=float)
    idx = np.searchsorted(t, v.fillna(np.inf).to_numpy(dtype=float), side="right")
    if best == "high":
        idx = len(t) - idx
    out = pd.Series(
        pd.array([LETTERS[i] for i in np.clip(idx, 0, len(t))], dtype="string"),
        index=v.index,
    )
    out[v.isna()] = pd.NA
    return out


def potential_heat(
    gas: pd.Series, elec: pd.Series, fabric_factor: pd.Series
) -> pd.Series:
    """Per-household home energy under full fabric + heat-pump deployment.

    Mirrors the heat side of ``scenarios.scenario_energy`` at
    ``("fabric+hp", u_heat=1)``, taking the fabric factor as an argument so the
    caller can supply a missing-EPC fallback (``test_nepi_score`` asserts the
    equivalence on complete rows). The EV side is a plain ratio applied in
    ``build_scores``.

    Parameters
    ----------
    gas, elec : pandas.Series
        Baseline per-household metered gas and electricity (kWh/yr).
    fabric_factor : pandas.Series
        EPC potential/current intensity ratio, missing values already filled.

    Returns
    -------
    pandas.Series
        Home energy (kWh/household/yr) under full deployment.
    """
    return gas * fabric_factor * (BOILER_EFF / COP) + elec


def load_or_freeze_bands(frame: pd.DataFrame) -> dict:
    """Load the frozen band thresholds, computing and writing them if absent.

    Parameters
    ----------
    frame : pandas.DataFrame
        Scoreable per-OA frame with ``rate``, ``total_kwh_hh``,
        ``access_walk`` and ``total_hh``.

    Returns
    -------
    dict
        The bands document (version, weighting, thresholds per quantity).
    """
    if BANDS_PATH.exists():
        bands = json.loads(BANDS_PATH.read_text())
        if bands["version"] == BANDS_VERSION:
            print(f"  Bands: {bands['version']} (frozen, {BANDS_PATH.name})")
            return bands
        print(f"  Bands: {bands['version']} superseded — re-freezing {BANDS_VERSION}")
    hh = _num(frame["total_hh"])
    bands = {
        "version": BANDS_VERSION,
        "n_bands": N_BANDS,
        "weighting": "households",
        "note": (
            "Household-weighted septiles of the 2021 national distribution; "
            "A is the best band. Computed once and frozen (score_spec.md)."
        ),
        "thresholds": {
            "rate": weighted_thresholds(frame["rate"], hh),
            "energy": weighted_thresholds(frame["total_kwh_hh"], hh),
            "access": weighted_thresholds(frame["access_walk"], hh),
        },
    }
    BANDS_PATH.write_text(json.dumps(bands, indent=2) + "\n")
    print(f"  Bands: {BANDS_VERSION} computed and frozen → {BANDS_PATH.name}")
    return bands


def build_scores() -> pd.DataFrame:
    """Assemble the per-OA NEPI score frame (letters, lever inputs, flags)."""
    df = load_and_aggregate()
    net = pd.read_parquet(NET_CACHE, columns=["net_total_1600", "net_amen"])
    df = df.merge(net, left_on="OA21CD", right_index=True, how="left", validate="m:1")

    hh = _num(df["total_hh"])
    gas = (_num(df["oa_gas_mean_kwh"]) * _num(df["oa_gas_num_meters"])).fillna(0) / hh
    elec = (_num(df["oa_elec_mean_kwh"]) * _num(df["oa_elec_num_meters"])).fillna(
        0
    ) / hh
    travel = _num(df["transport_kwh_per_hh_total_est"])

    fabric = _fabric_factor(df)
    ev_ratio = KWH_PER_MILE_EV / fleet_intensity_kwh_per_mile(df)

    out = pd.DataFrame(
        {
            "OA21CD": df["OA21CD"],
            "LSOA21CD": df["LSOA21CD"],
            "LAD22CD": df["LAD22CD"],
            "dominant_type": df["dominant_type"].astype(str),
            "total_hh": hh,
            "avg_hh_size": _num(df["avg_hh_size"]),
            "floor_area_m2": _num(df["oa_median_floor_area_m2"]),
            "gas_kwh_hh": gas,
            "elec_kwh_hh": elec,
            "travel_kwh_hh": travel,
            "fabric_factor": fabric.fillna(1.0),
            "ev_ratio": ev_ratio,
            "access_walk": _num(df["net_total_1600"]),
            "access_catchment": _num(df["net_amen"]),
            "flag_no_epc": fabric.isna(),
            "flag_low_meters": _num(df["oa_elec_num_meters"]) < LOW_METER_RATIO * hh,
        }
    )

    scoreable = out["access_catchment"].notna() & (out["travel_kwh_hh"] > 0)
    n_dropped = int((~scoreable).sum())
    if n_dropped:
        print(f"  {n_dropped:,} OAs unscoreable (absent from the network cache).")
    out = out[scoreable].copy()

    out["total_kwh_hh"] = out["gas_kwh_hh"] + out["elec_kwh_hh"] + out["travel_kwh_hh"]
    out["rate"] = out["access_catchment"] / out["total_kwh_hh"]

    heat_pot = potential_heat(
        out["gas_kwh_hh"], out["elec_kwh_hh"], out["fabric_factor"]
    )
    travel_pot = out["travel_kwh_hh"] * out["ev_ratio"]
    out["total_kwh_hh_potential"] = heat_pot + travel_pot
    out["rate_potential"] = out["access_catchment"] / out["total_kwh_hh_potential"]

    bands = load_or_freeze_bands(out)
    t = bands["thresholds"]
    out["letter_rate"] = assign_letters(out["rate"], t["rate"], best="high")
    out["letter_rate_potential"] = assign_letters(
        out["rate_potential"], t["rate"], best="high"
    )
    out["letter_energy"] = assign_letters(out["total_kwh_hh"], t["energy"], best="low")
    out["letter_energy_potential"] = assign_letters(
        out["total_kwh_hh_potential"], t["energy"], best="low"
    )
    out["letter_access"] = assign_letters(out["access_walk"], t["access"], best="high")
    return out


def _letter_shares(frame: pd.DataFrame, col: str) -> str:
    hh = _num(frame["total_hh"])
    total = float(hh.sum())
    parts = []
    for letter in LETTERS:
        share = float(hh[frame[col] == letter].sum()) / total * 100
        parts.append(f"{letter} {share:4.1f}%")
    return "  ".join(parts)


def main() -> None:
    """Build the score frame, print the summary and write the parquet."""
    out = build_scores()

    print(f"\n  NEPI score — {len(out):,} OAs, bands {BANDS_VERSION}")
    bands = json.loads(BANDS_PATH.read_text())
    for key, unit in (
        ("rate", "amenities/kWh"),
        ("energy", "kWh/dwelling/yr"),
        ("access", "amenities on foot"),
    ):
        cuts = "  ".join(f"{x:,.3g}" for x in bands["thresholds"][key])
        print(f"    {key:<7s} thresholds ({unit}): {cuts}")

    print("\n  Letter shares (of households):")
    for label, col in (
        ("rate, current", "letter_rate"),
        ("rate, potential", "letter_rate_potential"),
        ("energy, current", "letter_energy"),
        ("energy, potential", "letter_energy_potential"),
        ("access", "letter_access"),
    ):
        print(f"    {label:<18s} {_letter_shares(out, col)}")

    moved = (out["letter_rate_potential"] != out["letter_rate"]).mean() * 100
    print(f"\n  OAs changing rate letter under full deployment: {moved:.1f}%")
    for flag in ("flag_no_epc", "flag_low_meters"):
        print(f"  {flag}: {int(out[flag].sum()):,} OAs")

    out.to_parquet(OUT_PATH, index=False)
    print(f"\n  → {OUT_PATH}")


if __name__ == "__main__":
    main()
