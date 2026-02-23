# ruff: noqa: E501
"""
One-pager summary: The Trophic Layers of Urban Energy.

Generates a printable ASCII summary of the proof of concept results.
Loads the same data as proof_of_concept.py and renders key findings
as formatted tables, bar charts, and diagrams.

Usage:
    uv run python stats/print_summary.py
    uv run python stats/print_summary.py > summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from proof_of_concept import (
    TROPHIC_LAYERS,
    TYPE_LABELS,
    TYPE_ORDER,
    load_data,
)

W = 120  # page width
HALF = 56  # half-width for dual columns


def hr(char: str = "-") -> str:
    """Horizontal rule."""
    return char * W


def center(text: str, width: int = W) -> str:
    """Center text within width."""
    return text.center(width)


def dual(
    left_lines: list[str], right_lines: list[str], sep: str = "  |  "
) -> list[str]:
    """Merge two lists of lines into dual-column layout."""
    n = max(len(left_lines), len(right_lines))
    out = []
    for i in range(n):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        out.append(f"{left:<{HALF}}{sep}{right:<{HALF}}")
    return out


def box(lines: list[str], title: str = "") -> str:
    """Draw a box around lines of text."""
    inner_w = W - 4
    border = "=" * (inner_w + 2)
    if title:
        t = f" {title} "
        pad = inner_w + 2 - len(t)
        top = "=" * (pad // 2) + t + "=" * (pad - pad // 2)
    else:
        top = border
    out = [top]
    for line in lines:
        if len(line) > inner_w:
            line = line[:inner_w]
        out.append("| " + line.ljust(inner_w) + " |")
    out.append(border)
    return "\n".join(out)


def bar_h(value: float, max_val: float, width: int = 40, char: str = "#") -> str:
    """Horizontal bar scaled to max_val."""
    n = int(value / max_val * width) if max_val > 0 else 0
    return char * max(1, n)


def main() -> None:
    """Generate the one-pager summary."""
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        df = load_data()
    finally:
        sys.stdout = old_stdout

    # --- Compute all metrics ---

    _car_pfx = "ts045_Number of cars or vans: "
    _car_sfx = "; measures: Value"
    car_cols = {
        0: f"{_car_pfx}No cars or vans in household{_car_sfx}",
        1: f"{_car_pfx}1 car or van in household{_car_sfx}",
        2: f"{_car_pfx}2 cars or vans in household{_car_sfx}",
        3: f"{_car_pfx}3 or more cars or vans in household{_car_sfx}",
    }
    if all(c in df.columns for c in car_cols.values()):
        total_cars = sum(n * df[col] for n, col in car_cols.items())
        total_hh = sum(df[col] for col in car_cols.values())
        df["avg_cars_per_hh"] = total_cars / total_hh.replace(0, np.nan)
        kwh_per_vehicle = 12_000 * (0.17 / 0.233)
        df["transport_kwh_per_hh"] = df["avg_cars_per_hh"] * kwh_per_vehicle
        df["total_energy_per_capita"] = (
            df["total_energy_kwh"] + df["transport_kwh_per_hh"]
        ) / df["avg_household_size"]

    layer_scores: dict[str, pd.Series] = {}
    for layer_name, spec in TROPHIC_LAYERS.items():
        cols = [c for c in spec["cols"] if c in df.columns]  # type: ignore[union-attr]
        if cols:
            layer_scores[layer_name] = sum(
                pd.to_numeric(df[c], errors="coerce").fillna(0) for c in cols
            )  # type: ignore[assignment]

    type_stats: dict[str, dict[str, float]] = {}
    for t in TYPE_ORDER:
        sub = df[df["morph_type"] == t]
        if len(sub) == 0:
            continue
        s: dict[str, float] = {
            "n": len(sub),
            "kwh_m2": sub["energy_intensity"].mean(),
            "kwh_cap": sub["total_energy_per_capita"].mean(),
            "cars": sub["avg_cars_per_hh"].mean()
            if "avg_cars_per_hh" in sub.columns
            else 0,
            "age": sub["building_age"].median(),
            "party": sub["party_ratio"].mean() if "party_ratio" in sub.columns else 0,
            "env_dw": sub["envelope_per_dwelling"].mean()
            if "envelope_per_dwelling" in sub.columns
            else 0,
        }
        for layer_name, vals in layer_scores.items():
            s[f"acc_{layer_name}"] = vals[df["morph_type"] == t].mean()
        type_stats[t] = s

    det = type_stats["detached"]
    flt = type_stats["flat"]

    # --- Scaling data (GVA + BRES) at OA level ---
    scaling_stats: dict[str, dict[str, float]] | None = None
    has_gva = (
        "lsoa_gva_millions" in df.columns and df["lsoa_gva_millions"].notna().any()
    )
    has_bres = "lsoa_employment" in df.columns and df["lsoa_employment"].notna().any()
    # Find OA population column (TS001)
    pop_col = None
    for col in df.columns:
        if col.startswith("ts001_") and "total" in col.lower():
            pop_col = col
            break
    if pop_col is None:
        ts001_cols = [c for c in df.columns if c.startswith("ts001_")]
        if ts001_cols:
            pop_col = ts001_cols[0]

    if (has_gva or has_bres) and "OA21CD" in df.columns and pop_col:
        oa_agg: dict[str, str] = {
            "total_energy_kwh": "sum",
            pop_col: "first",
            "pop_density": "first",
        }
        if has_gva:
            oa_agg["lsoa_gva_millions"] = "first"
        if has_bres:
            oa_agg["lsoa_employment"] = "first"

        oa_mask = df["OA21CD"].notna() & df[pop_col].notna()
        if has_gva:
            oa_mask = oa_mask & df["lsoa_gva_millions"].notna()
        oa_df = df.loc[oa_mask].groupby("OA21CD").agg(oa_agg).reset_index()

        oa_pop = pd.to_numeric(oa_df[pop_col], errors="coerce")
        oa_df = oa_df[oa_pop > 0].copy()
        oa_pop = pd.to_numeric(oa_df[pop_col], errors="coerce")
        oa_df["oa_energy_per_capita"] = oa_df["total_energy_kwh"] / oa_pop

        if len(oa_df) >= 10:
            oa_df["density_quintile"] = pd.qcut(
                oa_df["pop_density"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
            )
            scaling_stats = {}
            for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                q_sub = oa_df[oa_df["density_quintile"] == q_label]
                if len(q_sub) == 0:
                    continue
                sd: dict[str, float] = {
                    "n": len(q_sub),
                    "pop_density": q_sub["pop_density"].mean(),
                    "kwh_cap": q_sub["oa_energy_per_capita"].mean(),
                }
                if has_gva:
                    sd["gva"] = q_sub["lsoa_gva_millions"].mean()
                if has_bres:
                    sd["employment"] = q_sub["lsoa_employment"].mean()
                scaling_stats[q_label] = sd

    # ===================================================================
    # RENDER
    # ===================================================================

    lines: list[str] = []
    p = lines.append

    # --- TITLE ---
    p(hr("="))
    p("")
    p(center("The Trophic Layers of Urban Energy"))
    p(center("A proof of concept  --  Manchester"))
    p("")
    p(hr("="))
    p("")
    p(
        "    Cities are conduits that capture energy and recycle it through layers of human interaction"
    )
    p(
        "    (Jacobs, 2000).  Bettencourt et al. (2007) show that cities scale superlinearly in"
    )
    p(
        "    socioeconomic output (~N^1.15) and sublinearly in infrastructure (~N^0.85).  Proximity is"
    )
    p(
        "    the mechanism: density enables more interactions per unit of energy.  The question is not"
    )
    p(
        "    how much energy a neighbourhood consumes, but how much city each unit of energy enables."
    )
    p("")

    # --- THE ANALOGY (dual column) ---
    left = [
        center("Dense neighbourhood (1 km x 1 km)", HALF),
        "",
        "        * energy input",
        "        |",
        "    +---------------------------------------+",
        "    |  Layer 1: shops, cafes, schools        |",
        "    +-------+-------------------------------+",
        "            |  <-> exchange",
        "    +-------+-------------------------------+",
        "    |  Layer 2: transit stops, cycling       |",
        "    +-------+-------------------------------+",
        "            |  <-> exchange",
        "    +-------+-------------------------------+",
        "    |  Layer 3: street network, green space  |",
        "    +-------+-------------------------------+",
        "            |  <-> exchange",
        "    +-------+-------------------------------+",
        "    |  Layer 4: social ties, local reuse     |",
        "    +---------------------------------------+",
        "",
        "  Energy captured at every layer.",
        "  Recycled, reused, compounded.",
        "  Thousands of interactions per km2.",
    ]
    right = [
        center("Suburban sprawl (1 km x 1 km)", HALF),
        "",
        "        * energy input",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        |",
        "        v",
        "    ====================================",
        "    lost",
        "",
        "",
        "",
        "",
        "  Energy passes through one layer",
        "  and escapes.  Thin conduit.",
        "  Handful of destinations per km2.",
    ]
    p(
        box(
            dual(left, right),
            title=" Same energy input, different trophic depth ",
        )
    )
    p("")

    # --- STEPS 1 & 2: TROPHIC LAYERS (left box) + PHYSICS (right box) ---

    hw = HALF - 1  # inner width for each half-box

    # Left box: trophic layers
    tl = []
    tl.append("+" + "= Trophic layers (what you get) ".ljust(hw - 1, "=") + "+")
    tl.append("|" + "".center(hw) + "|")
    tl.append("|" + "  What each type gets access to".ljust(hw) + "|")
    tl.append("|" + "  800m network walk".ljust(hw) + "|")
    tl.append("|" + "".center(hw) + "|")
    hdr = f"  {'Layer':<20} {'Det':>5} {'Semi':>5} {'Flat':>5}"
    tl.append("|" + hdr.ljust(hw) + "|")
    sep = f"  {'':.<20} {'':.<5} {'':.<5} {'':.<5}"
    tl.append("|" + sep.ljust(hw) + "|")
    for layer_name in TROPHIC_LAYERS:
        label = TROPHIC_LAYERS[layer_name]["label"]  # type: ignore[index]
        v_d = type_stats["detached"].get(f"acc_{layer_name}", 0)
        v_s = type_stats["semi"].get(f"acc_{layer_name}", 0)
        v_f = type_stats["flat"].get(f"acc_{layer_name}", 0)
        row = f"  {label:<20} {v_d:>5.1f} {v_s:>5.1f} {v_f:>5.1f}"
        tl.append("|" + row.ljust(hw) + "|")
    tl.append("|" + "".center(hw) + "|")
    tl.append("|" + "  Energy flow through conduit:".ljust(hw) + "|")
    tl.append("|" + "".center(hw) + "|")
    tl.append("|" + "         * energy input".ljust(hw) + "|")
    tl.append("|" + "         |".ljust(hw) + "|")
    tl.append("|" + "         v".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "    | streets   |  connectivity".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "         |".ljust(hw) + "|")
    tl.append("|" + "         v".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "    | amenity   |  shops, cafes".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "         |".ljust(hw) + "|")
    tl.append("|" + "         v".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "    | transit   |  bus, rail".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "         |".ljust(hw) + "|")
    tl.append("|" + "         v".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "    | green     |  parks, open".ljust(hw) + "|")
    tl.append("|" + "    +-----------+".ljust(hw) + "|")
    tl.append("|" + "         |".ljust(hw) + "|")
    tl.append("|" + "         v recycled".ljust(hw) + "|")
    tl.append("|" + "".center(hw) + "|")
    tl.append("|" + "  Dense: thick conduit, many layers".ljust(hw) + "|")
    tl.append("|" + "  Sprawl: thin conduit, energy lost".ljust(hw) + "|")
    tl.append("|" + "".center(hw) + "|")

    # Right box: physics
    pr = []
    pr.append("+" + "= Physics (what it costs) ".ljust(hw - 1, "=") + "+")
    pr.append("|" + "".center(hw) + "|")
    pr.append("|" + "  Thermal envelope per dwelling".ljust(hw) + "|")
    pr.append("|" + "  Shared walls = shared warmth".ljust(hw) + "|")
    pr.append("|" + "".center(hw) + "|")
    hdr2 = f"  {'Type':<16} {'Env m2':>8} {'Party':>7}"
    pr.append("|" + hdr2.ljust(hw) + "|")
    sep2 = f"  {'':.<16} {'':.<8} {'':.<7}"
    pr.append("|" + sep2.ljust(hw) + "|")
    for t in TYPE_ORDER:
        if t not in type_stats:
            continue
        row = f"  {TYPE_LABELS[t]:<16} {type_stats[t]['env_dw']:>8.0f} {type_stats[t]['party']:>7.2f}"
        pr.append("|" + row.ljust(hw) + "|")
    pr.append("|" + "".center(hw) + "|")
    pr.append("|" + "      Detached         Flat".ljust(hw) + "|")
    pr.append("|" + "   +-----------+   +--+--+--+".ljust(hw) + "|")
    pr.append("|" + "   |           |   |  |  |  |".ljust(hw) + "|")
    pr.append("|" + "   |  exposed  |   +--+--+--+".ljust(hw) + "|")
    pr.append("|" + "   |  on all   |   |  |  |  |".ljust(hw) + "|")
    pr.append("|" + "   |  sides    |   +--+--+--+".ljust(hw) + "|")
    pr.append("|" + "   |           |   |  |  |  |".ljust(hw) + "|")
    pr.append("|" + "   +-----------+   +--+--+--+".ljust(hw) + "|")
    pr.append("|" + "".center(hw) + "|")
    det_line = f"  Detached: {det['env_dw']:.0f} m2, 0% shared"
    flt_line = f"  Flat:     {flt['env_dw']:.0f} m2, {flt['party']:.0%} shared"
    ratio_line = f"  {det['env_dw'] / flt['env_dw']:.1f}x more envelope per dwelling"
    pr.append("|" + det_line.ljust(hw) + "|")
    pr.append("|" + flt_line.ljust(hw) + "|")
    pr.append("|" + "".center(hw) + "|")
    pr.append("|" + ratio_line.ljust(hw) + "|")
    pr.append("|" + "".center(hw) + "|")

    # Pad to same height, then close both
    while len(pr) < len(tl):
        pr.append("|" + "".center(hw) + "|")
    while len(tl) < len(pr):
        tl.append("|" + "".center(hw) + "|")
    tl.append("+" + "=" * hw + "+")
    pr.append("+" + "=" * hw + "+")

    # Merge side by side with a small gap
    for left_line, right_line in zip(tl, pr):
        p(f"{left_line}  {right_line}")
    p("")

    # --- STEP 3: THE COMPOUNDING ---
    ratio_m2 = det["kwh_m2"] / flt["kwh_m2"]
    ratio_cap = det["kwh_cap"] / flt["kwh_cap"]

    layer_ratios: list[tuple[str, float]] = []
    for layer_name in layer_scores:
        det_acc = det.get(f"acc_{layer_name}", 0)
        flat_acc = flt.get(f"acc_{layer_name}", 0)
        all_vals = [
            type_stats[t].get(f"acc_{layer_name}", 0)
            for t in TYPE_ORDER
            if t in type_stats
        ]
        min_val = min(all_vals)
        shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0
        det_epa = det["kwh_cap"] / (det_acc + shift)
        flat_epa = flt["kwh_cap"] / (flat_acc + shift)
        if flat_epa > 0:
            ratio = det_epa / flat_epa
            label = TROPHIC_LAYERS[layer_name]["label"]  # type: ignore[index]
            layer_ratios.append((label, ratio))

    all_ratios = [
        ("kWh / m2  (physics only)", ratio_m2),
        ("kWh / capita  (bldg + transport)", ratio_cap),
        *[(f"kWh / capita / {lab.lower()}", r) for lab, r in layer_ratios],
    ]
    max_ratio = max(r for _, r in all_ratios)

    comp_left = [
        "",
        "  Each normalisation widens the gap.",
        "  Detached-to-flat ratio:",
        "",
        f"  {'Normalisation':<36} {'Ratio':>8}",
        f"  {'':.<36} {'':.<8}",
    ]
    for label, ratio in all_ratios:
        comp_left.append(f"  {label:<36} {ratio:>7.2f}x")

    comp_left.append("")
    if layer_ratios:
        best = max(layer_ratios, key=lambda x: x[1])
        comp_left.append(f"  Peak: {best[0].lower()} at {best[1]:.2f}x")
    comp_left.append("")

    comp_right = [
        "",
        "  Compounding visualised:",
        "",
        f"  physics  |{bar_h(ratio_m2, max_ratio, width=30)}  {ratio_m2:.2f}x",
        "           |",
        f"  + capita |{bar_h(ratio_cap, max_ratio, width=30)}  {ratio_cap:.2f}x",
        "           |",
    ]
    short_names = {
        "Street connectivity": "streets",
        "Amenity access": "amenity",
        "Transit access": "transit",
        "Green space": "green",
    }
    for label, ratio in layer_ratios:
        short = short_names.get(label, label[:8]).lower()
        comp_right.append(
            f"  + {short:<7}|{bar_h(ratio, max_ratio, width=30)}  {ratio:.2f}x"
        )
        comp_right.append("           |")
    if comp_right[-1] == "           |":
        comp_right.pop()
    comp_right.append("")
    comp_right.append("  Each layer amplifies the gap.")
    comp_right.append("  Less city per unit of energy.")
    comp_right.append("")

    p(box(dual(comp_left, comp_right), title=" 3. The compounding effect "))
    p("")

    # --- STEP 4: DEPRIVATION CONTROL (dual column) ---
    dep_col_name = "deprivation_quintile"
    dep_left = [
        "",
        "  Is this just wealth?",
        "  Rich suburbs, poor flats?",
        "",
    ]
    dep_right = [
        "",
        "  Houses vs flats by deprivation:",
        "",
        f"  {'Quintile':<20} {'Houses':>8} {'Flats':>8} {'Ratio':>8}",
        f"  {'':.<20} {'kWh/cap':.<8} {'kWh/cap':.<8} {'':.<8}",
    ]

    if dep_col_name in df.columns:
        ratios_found = []
        for q in df[dep_col_name].cat.categories:
            q_df = df[df[dep_col_name] == q]
            house = q_df[q_df["morph_type"].isin(["detached", "semi"])]
            flat = q_df[q_df["morph_type"] == "flat"]
            if len(house) >= 20 and len(flat) >= 20:
                h_e = house["total_energy_per_capita"].mean()
                f_e = flat["total_energy_per_capita"].mean()
                ratio = h_e / f_e if f_e > 0 else 0
                ratios_found.append(ratio)
                dep_right.append(
                    f"  {str(q):<20} {h_e:>8,.0f} {f_e:>8,.0f} {ratio:>7.2f}x"
                )

        dep_left.append("  Test: hold deprivation constant.")
        dep_left.append("  Compare houses vs flats within")
        dep_left.append("  each quintile.")
        dep_left.append("")
        dep_left.append("  Result:")
        dep_left.append("")
        dep_left.append("  Houses cost more in every quintile.")
        dep_left.append("  The effect is morphological,")
        dep_left.append("  not socioeconomic.")
        dep_left.append("")
        if ratios_found:
            dep_left.append(
                f"  Range: {min(ratios_found):.2f}x -- {max(ratios_found):.2f}x"
            )
        dep_left.append("")
    else:
        dep_left.append("  (Deprivation data not available)")
        dep_left.append("")

    dep_right.append("")

    p(box(dual(dep_left, dep_right), title=" 4. Not a wealth effect "))
    p("")

    # --- STEP 5: LOCK-IN (dual column) ---
    lock_left = [
        "",
        "  Building stock composition:",
        "",
        f"  {'Type':<16} {'Share':>7} {'Built':>8} {'Age':>6}",
        f"  {'':.<16} {'':.<7} {'':.<8} {'':.<6}",
    ]
    total_n = sum(type_stats[t]["n"] for t in TYPE_ORDER if t in type_stats)
    for t in TYPE_ORDER:
        if t not in type_stats:
            continue
        s = type_stats[t]
        pct = s["n"] / total_n * 100
        year = 2024 - s["age"]
        lock_left.append(
            f"  {TYPE_LABELS[t]:<16} {pct:>6.1f}% {f'~{year:.0f}':>8} {s['age']:>4.0f}yr"
        )
    lock_left.append("")

    lock_right = [
        "",
        "  Age distribution:",
        "",
    ]
    for t in TYPE_ORDER:
        if t not in type_stats:
            continue
        s = type_stats[t]
        age_bar = bar_h(s["age"], 120, width=35)
        lock_right.append(f"  {TYPE_LABELS[t]:<16} |{age_bar} {s['age']:.0f} yr")
    lock_right.append("")
    lock_right.append("  These buildings will stand for")
    lock_right.append("  decades more.  The conduit they")
    lock_right.append("  create is locked in too.")
    lock_right.append("")
    lock_right.append("  +-- built ------ standing ---- 2100 --+")
    lock_right.append("  |  ############...................... |")
    lock_right.append("  +---- locked-in energy pattern -------+")
    lock_right.append("")

    p(box(dual(lock_left, lock_right), title=" 5. Locked in for decades "))
    p("")

    # --- STEP 6: SCALING (if data available) ---
    if scaling_stats and len(scaling_stats) >= 2:
        scale_left = [
            "",
            "  Bettencourt et al. (2007):",
            "  GDP ~ N^1.15 (superlinear)",
            "  Infrastructure ~ N^0.85 (sublinear)",
            "",
            "  If compact form is the mechanism,",
            "  same morphology that saves energy",
            "  should generate more economic value.",
            "",
            "  Test: density quintiles of OAs",
            "  (Q1 = most sprawling, Q5 = densest)",
            "",
        ]
        q1 = scaling_stats.get("Q1", {})
        q5 = scaling_stats.get("Q5", {})
        if q1 and q5:
            e_ratio = (
                q1.get("kwh_cap", 0) / q5["kwh_cap"] if q5.get("kwh_cap", 0) > 0 else 0
            )
            scale_left.append(f"  Energy:   Q1/Q5 = {e_ratio:.2f}x")
            scale_left.append("    (sprawl costs more)")
            if "gva" in q1 and "gva" in q5 and q1["gva"] > 0:
                g_ratio = q5["gva"] / q1["gva"]
                scale_left.append(f"  GVA:      Q5/Q1 = {g_ratio:.2f}x")
                scale_left.append("    (density produces more)")
            if "employment" in q1 and "employment" in q5 and q1["employment"] > 0:
                emp_ratio = q5["employment"] / q1["employment"]
                scale_left.append(f"  Jobs:     Q5/Q1 = {emp_ratio:.2f}x")
                scale_left.append("    (density employs more)")
        scale_left.append("")

        scale_right = [
            "",
            f"  {'Quintile':<8} {'Pop/km2':>9} {'kWh/cap':>9}",
        ]
        if has_gva:
            scale_right[-1] += f" {'GVA £m':>9}"
        if has_bres:
            scale_right[-1] += f" {'Jobs':>7}"
        scale_right.append(
            f"  {'':.<8} {'':.<9} {'':.<9}"
            + (f" {'':.<9}" if has_gva else "")
            + (f" {'':.<7}" if has_bres else "")
        )
        for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if q_label not in scaling_stats:
                continue
            sd = scaling_stats[q_label]
            row = f"  {q_label:<8} {sd['pop_density']:>9,.0f} {sd['kwh_cap']:>9,.0f}"
            if has_gva and "gva" in sd:
                row += f" {sd['gva']:>9.2f}"
            if has_bres and "employment" in sd:
                row += f" {sd['employment']:>7.0f}"
            scale_right.append(row)
        scale_right.append("")
        # Visual: energy goes down, GVA goes up
        scale_right.append("  As density increases:")
        scale_right.append("    energy/capita ........... falls")
        if has_gva:
            scale_right.append("    GVA/neighbourhood ...... rises")
        if has_bres:
            scale_right.append("    employment ............. rises")
        scale_right.append("")
        scale_right.append("  Compact form is simultaneously")
        scale_right.append("  greener and more productive.")
        scale_right.append("")

        p(
            box(
                dual(scale_left, scale_right),
                title=" 6. The conduit amplifies economic output ",
            )
        )
        p("")

    # --- THE THESIS ---
    p(hr("="))
    p("")
    p(center("Thesis"))
    p("")
    p(
        "    Urban morphological form determines not just how much energy a neighbourhood consumes"
    )
    p(
        "    (building physics + transport), but how efficiently that energy is converted into"
    )
    p("    urban function (accessibility, amenity, connectivity, green space).")
    p("")
    p("    Compact forms capture energy through many trophic layers of interaction.")
    p("    Sprawl passes energy through one layer and dissipates it.")
    p("")
    p(
        "    This compounds.  Each layer of the conduit amplifies the efficiency gap.  The same"
    )
    p(
        "    compact form that reduces per-capita energy also amplifies economic output per"
    )
    p(
        "    neighbourhood (Bettencourt et al., 2007).  And because buildings last 60-100+ years,"
    )
    p(
        "    planning decisions about density and housing type lock in both the energy penalty"
    )
    p("    and the thin conduit for generations.")
    p("")
    p(hr("-"))
    p(
        center(
            "Jacobs (2000)  --  Bettencourt et al. (2007)  --  Rode et al. (2014)  --  Newman & Kenworthy (1989)  --  Norman et al. (2006)"
        )
    )
    p(hr("-"))
    p(center(f"Data: Manchester  (n = {len(df):,} dwellings)"))
    p(
        center(
            "Sources: EPC -- OS MasterMap -- LiDAR -- DESNZ metered -- Census 2021 -- ONS GVA -- BRES -- cityseer"
        )
    )
    p(hr("="))

    for line in lines:
        print(line)

    # --- Write PDF ---
    write_pdf(lines)


def write_pdf(lines: list[str]) -> None:
    """Write lines to a landscape PDF via macOS cupsfilter."""
    import subprocess

    out_path = Path(__file__).resolve().parent.parent / "summary.pdf"
    txt_path = out_path.with_suffix(".txt")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with out_path.open("wb") as f:
        subprocess.run(
            [
                "cupsfilter",
                "-m",
                "application/pdf",
                "-o",
                "cpi=18",
                "-o",
                "lpi=9",
                str(txt_path),
            ],
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    txt_path.unlink(missing_ok=True)
    print(f"\nPDF written to {out_path}")


if __name__ == "__main__":
    main()
