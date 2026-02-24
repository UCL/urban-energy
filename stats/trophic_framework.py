"""
Rigorous formalisation of the trophic framework.

Replaces the ad hoc ratio-based compounding test (PoC Step 4) with three
defensible approaches:

    A. City index construction
       - PCA composite: data-driven "Urban Service Intensity" from cityseer
         accessibility metrics. Gives a single, reproducible index.
       - Shannon entropy: functional diversity of amenity mix within 800m.
         Maps directly to the trophic depth concept — how many distinct
         types of interaction the catchment supports.

    B. Denominator progression with bootstrap CIs
       Formalises the "widening gap" claim. At each normalisation level
       (kWh/m² → kWh/capita → kWh/capita/city), bootstrap the Detached/Flat
       ratio and test whether successive ratios are significantly different.
       This replaces eyeballing ratios with statistical inference.

    C. Interaction regression
       Tests whether morphological type *moderates* the energy→city
       relationship. The model:
           city_pca ~ log(energy_per_capita) * morph_type + controls
       If the interaction term is positive for compact types, they convert
       energy to city more efficiently. This is the compounding claim
       stated as a testable regression coefficient.

    D. Bettencourt scaling test
       At OA level, test whether compact morphology predicts higher GVA
       than expected from population alone (log-log scaling residual).

Usage:
    uv run python stats/trophic_framework.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

# Import data loading and constants from existing PoC
sys.path.insert(0, str(Path(__file__).parent))
from proof_of_concept import (
    TYPE_LABELS,
    TYPE_ORDER,
    _sigstars,
    aggregate_to_oa,
    load_data,
)

# Cityseer columns: gravity-weighted amenity counts at 800m network distance
AMENITY_COLS = [
    "cc_fsa_restaurant_800_wt",
    "cc_fsa_pub_800_wt",
    "cc_fsa_takeaway_800_wt",
    "cc_fsa_other_800_wt",
    "cc_bus_800_wt",
    "cc_rail_800_wt",
    "cc_greenspace_800_wt",
]

# Full accessibility set (amenities + street network centrality)
ACCESS_COLS = ["cc_harmonic_800"] + AMENITY_COLS


# ---------------------------------------------------------------------------
# A. City Index Construction
# ---------------------------------------------------------------------------


def construct_city_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct composite city indices from cityseer metrics.

    Two complementary measures:

    PCA (Urban Service Intensity)
        First principal component of standardised accessibility metrics.
        Captures the dominant axis of variation — a general urbanity factor.
        Advantages: data-driven weights, single score, high variance explained.
        Limitation: may collapse to a density proxy. Check loadings.

    Shannon entropy (Trophic Depth)
        H = -Σ pᵢ·ln(pᵢ) over amenity type proportions within 800m.
        Captures functional diversity: a catchment with restaurants AND pubs
        AND transit AND green space has higher H than one dominated by a
        single type. Maps directly to the ecological analogy.
        Advantages: interpretable (nats), clear theoretical justification.
        Limitation: ignores total volume (a desert with one of each = high H).

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with cityseer accessibility columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with ``city_pca`` and ``city_entropy`` columns added.
    """
    print("\n" + "=" * 70)
    print("A. CITY INDEX CONSTRUCTION")
    print("=" * 70)

    # --- PCA on standardised accessibility metrics ---
    acc_cols = [c for c in ACCESS_COLS if c in df.columns]
    if len(acc_cols) < 3:
        print("  SKIP: Fewer than 3 accessibility columns available")
        return df

    for c in acc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid_mask = df[acc_cols].notna().all(axis=1)
    X_raw = df.loc[valid_mask, acc_cols].values

    # Drop zero-variance columns (e.g. no rail in small towns)
    col_stds = X_raw.std(axis=0)
    nonzero_var = col_stds > 0
    if not nonzero_var.all():
        dropped = [acc_cols[i] for i, v in enumerate(nonzero_var) if not v]
        print(f"  Dropping zero-variance columns: {dropped}")
        acc_cols = [acc_cols[i] for i, v in enumerate(nonzero_var) if v]
        X_raw = X_raw[:, nonzero_var]
        col_stds = col_stds[nonzero_var]

    # Standardise
    X = (X_raw - X_raw.mean(axis=0)) / col_stds

    # Eigendecomposition of correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pc1_scores = X @ eigenvectors[:, 0]
    var_explained = eigenvalues[0] / eigenvalues.sum()

    # Orient PC1 so higher = more urban
    if "pop_density" in df.columns:
        corr_with_density = np.corrcoef(
            pc1_scores,
            df.loc[valid_mask, "pop_density"].values,
        )[0, 1]
        if corr_with_density < 0:
            pc1_scores = -pc1_scores
            eigenvectors[:, 0] = -eigenvectors[:, 0]

    df.loc[valid_mask, "city_pca"] = pc1_scores

    print("\n  PCA loadings on PC1 (Urban Service Intensity):")
    print(f"  Variance explained: {var_explained:.1%}")
    print(f"  {'Variable':<35} {'Loading':>8}")
    print("  " + "-" * 45)
    for col, loading in zip(acc_cols, eigenvectors[:, 0]):
        short = col.replace("cc_", "").replace("_800", "").replace("_wt", "")
        print(f"  {short:<35} {loading:>8.3f}")

    cum_var = np.cumsum(eigenvalues / eigenvalues.sum())
    print("\n  Eigenvalues (top 3):")
    for i in range(min(3, len(eigenvalues))):
        print(
            f"    PC{i + 1}: {eigenvalues[i]:.2f}"
            f" ({eigenvalues[i] / eigenvalues.sum():.1%},"
            f" cumulative {cum_var[i]:.1%})"
        )

    # --- Shannon entropy on amenity proportions ---
    amenity_cols = [c for c in AMENITY_COLS if c in df.columns]
    if len(amenity_cols) < 3:
        print("\n  SKIP: Too few amenity columns for entropy")
        return df

    amenity_vals = df[amenity_cols].clip(lower=0).fillna(0)
    total = amenity_vals.sum(axis=1)
    proportions = amenity_vals.div(total.replace(0, np.nan), axis=0)
    log_p = np.log(proportions.replace(0, 1))  # 0·ln(0) = 0
    df["city_entropy"] = -(proportions * log_p).sum(axis=1)
    df.loc[total == 0, "city_entropy"] = np.nan
    df["city_total_amenities"] = total

    h_max = np.log(len(amenity_cols))
    print("\n  Shannon entropy (trophic depth):")
    print(f"  Max possible H = ln({len(amenity_cols)}) = {h_max:.3f} nats")
    print(f"  {'Type':<18} {'Mean H':>8} {'H/Hmax':>8} {'Mean total':>12}")
    print("  " + "-" * 50)
    for t in TYPE_ORDER:
        sub = df[df["morph_type"] == t]
        mean_h = sub["city_entropy"].mean()
        mean_total = sub["city_total_amenities"].mean()
        print(
            f"  {TYPE_LABELS[t]:<18}"
            f" {mean_h:>8.3f}"
            f" {mean_h / h_max:>7.1%}"
            f" {mean_total:>12.1f}"
        )

    print("\n  Entropy = trophic DEPTH (diversity of functional layers).")
    print("  Total amenities = trophic MASS (volume of interactions).")
    print("  PCA combines both into a single urbanity axis.")

    return df


# ---------------------------------------------------------------------------
# B. Denominator Progression with Bootstrap CIs
# ---------------------------------------------------------------------------


def test_denominator_progression(
    df: pd.DataFrame,
    n_boot: int = 5_000,
    seed: int = 42,
) -> bool:
    """
    Formalise the compounding claim with bootstrap confidence intervals.

    At each normalisation level, bootstrap the Detached/Flat ratio of means
    and its 95% CI. Then test whether successive ratios differ significantly
    (CI of the difference excludes zero).

    Levels:
        1. kWh/m² — building physics
        2. kWh/capita — + transport, per person
        3. kWh/capita / USI — per unit of urban service (PCA)
        4. kWh/capita / H — per unit of trophic depth (entropy)

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with ``city_pca`` and ``city_entropy`` columns.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    bool
        True if at least one city-normalised ratio significantly exceeds
        the kWh/capita ratio.
    """
    print("\n" + "=" * 70)
    print("B. DENOMINATOR PROGRESSION (Bootstrap)")
    print("=" * 70)
    print("  Does each normalisation level significantly widen the gap?")

    rng = np.random.default_rng(seed)

    det = df[df["morph_type"] == "detached"].copy()
    flat = df[df["morph_type"] == "flat"].copy()

    if len(det) < 30 or len(flat) < 30:
        print("  SKIP: Too few observations")
        return False

    # --- Define normalisation levels ---
    levels: list[dict[str, object]] = []

    # Level 1: kWh/m²
    levels.append(
        {
            "label": "kWh/m²",
            "det_vals": det["energy_intensity"].dropna().values,
            "flat_vals": flat["energy_intensity"].dropna().values,
        }
    )

    # Level 2: kWh/capita (building + transport)
    levels.append(
        {
            "label": "kWh/capita",
            "det_vals": det["total_energy_per_capita"].dropna().values,
            "flat_vals": flat["total_energy_per_capita"].dropna().values,
        }
    )

    # Level 3: kWh/capita per unit PCA city index
    if "city_pca" in df.columns:
        min_pca = df["city_pca"].min()
        shift = abs(min_pca) + 1.0 if min_pca <= 0 else 0.0

        det_city = det["city_pca"].dropna() + shift
        flat_city = flat["city_pca"].dropna() + shift
        det_epc = det.loc[det_city.index, "total_energy_per_capita"]
        flat_epc = flat.loc[flat_city.index, "total_energy_per_capita"]

        levels.append(
            {
                "label": "kWh/cap / USI",
                "det_vals": (det_epc / det_city).dropna().values,
                "flat_vals": (flat_epc / flat_city).dropna().values,
            }
        )

    # Level 4: kWh/capita per unit entropy
    if "city_entropy" in df.columns:
        det_h = det["city_entropy"].dropna()
        flat_h = flat["city_entropy"].dropna()
        shift_h = 0.1  # entropy is non-negative; small shift avoids div by 0

        det_eph = det.loc[det_h.index, "total_energy_per_capita"] / (det_h + shift_h)
        flat_eph = flat.loc[flat_h.index, "total_energy_per_capita"] / (
            flat_h + shift_h
        )

        levels.append(
            {
                "label": "kWh/cap / entropy",
                "det_vals": det_eph.dropna().values,
                "flat_vals": flat_eph.dropna().values,
            }
        )

    # --- Bootstrap each level ---
    print(f"\n  {'Level':<24} {'Ratio':>7} {'95% CI':>18} {'N_det':>7} {'N_flat':>7}")
    print("  " + "-" * 68)

    boot_ratios_by_level: list[np.ndarray] = []

    for level in levels:
        d = level["det_vals"]
        f = level["flat_vals"]
        if len(d) < 10 or len(f) < 10:
            print(f"  {level['label']:<24} insufficient data")
            boot_ratios_by_level.append(np.full(n_boot, np.nan))
            continue

        boot = np.empty(n_boot)
        for b in range(n_boot):
            d_boot = rng.choice(d, size=len(d), replace=True)
            f_boot = rng.choice(f, size=len(f), replace=True)
            f_mean = f_boot.mean()
            boot[b] = d_boot.mean() / f_mean if f_mean > 0 else np.nan

        boot_ratios_by_level.append(boot)
        ci_lo, median, ci_hi = np.nanpercentile(boot, [2.5, 50, 97.5])
        print(
            f"  {level['label']:<24}"
            f" {median:>7.2f}x"
            f"  [{ci_lo:>6.2f}, {ci_hi:>6.2f}]"
            f" {len(d):>7,}"
            f" {len(f):>7,}"
        )

    # --- Test successive differences ---
    print("\n  Successive ratio differences:")
    compounding_found = False

    for i in range(1, len(boot_ratios_by_level)):
        if np.isnan(boot_ratios_by_level[i]).all():
            continue
        diff = boot_ratios_by_level[i] - boot_ratios_by_level[i - 1]
        ci_lo, median, ci_hi = np.nanpercentile(diff, [2.5, 50, 97.5])
        excludes_zero = ci_lo > 0

        label_from = levels[i - 1]["label"]
        label_to = levels[i]["label"]

        # City-normalised steps (i >= 2) that significantly widen the gap
        if excludes_zero and i >= 2:
            compounding_found = True

        print(
            f"    {label_from} -> {label_to}:"
            f"  delta = {median:>+.3f}  [{ci_lo:>+.3f}, {ci_hi:>+.3f}]"
            f"  {'SIGNIFICANT' if excludes_zero else 'not sig'}"
        )

    if compounding_found:
        print("\n  PASS: City normalisation significantly widens the gap.")
        print("  The accessibility step is not just the same effect restated.")
    else:
        print("\n  FAIL: No significant widening at the city-normalisation step.")

    return compounding_found


# ---------------------------------------------------------------------------
# C. Interaction Regression
# ---------------------------------------------------------------------------


def test_interaction_model(df: pd.DataFrame) -> bool:
    """
    Test whether morphological type moderates the energy -> city relationship.

    Model A (additive):
        city_pca ~ log(energy_per_capita) + morph_type + controls

    Model B (interaction):
        city_pca ~ log(energy_per_capita) * morph_type + controls

    If the interaction terms are jointly significant (Wald F-test),
    morphological type moderates how efficiently energy converts to city.
    A positive interaction for compact types means they get *more* city
    per marginal kWh — the compounding claim as a regression coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with ``city_pca`` column.

    Returns
    -------
    bool
        True if interaction terms are jointly significant at p < 0.05.
    """
    print("\n" + "=" * 70)
    print("C. INTERACTION MODEL: Does form moderate energy -> city?")
    print("=" * 70)

    if "city_pca" not in df.columns:
        print("  SKIP: No city_pca column")
        return False

    # Prepare model data
    model_cols = [
        "city_pca",
        "total_energy_per_capita",
        "morph_type",
        "log_floor_area",
        "building_age",
    ]
    model_df = df[model_cols].dropna().copy()
    model_df["log_energy_pc"] = np.log(
        model_df["total_energy_per_capita"].clip(lower=1)
    )
    model_df["morph_type"] = pd.Categorical(
        model_df["morph_type"],
        categories=TYPE_ORDER,
        ordered=True,
    )

    n = len(model_df)
    print(f"  N = {n:,}")
    if n < 100:
        print("  SKIP: Too few observations")
        return False

    ref = "Treatment(reference='detached')"

    # Model A: additive
    formula_a = (
        f"city_pca ~ log_energy_pc + C(morph_type, {ref})"
        " + log_floor_area + building_age"
    )
    model_a = smf.ols(formula_a, data=model_df).fit(cov_type="HC3")

    # Model B: with interaction
    formula_b = (
        f"city_pca ~ log_energy_pc * C(morph_type, {ref})"
        " + log_floor_area + building_age"
    )
    model_b = smf.ols(formula_b, data=model_df).fit(cov_type="HC3")

    # Wald test for joint significance of interaction terms
    interaction_params = [p for p in model_b.params.index if ":" in p]
    if not interaction_params:
        print("  SKIP: No interaction terms estimated")
        return False

    r_matrix = np.zeros((len(interaction_params), len(model_b.params)))
    for i, param in enumerate(interaction_params):
        j = list(model_b.params.index).index(param)
        r_matrix[i, j] = 1.0

    wald_test = model_b.wald_test(r_matrix, use_f=True)
    f_stat = float(wald_test.statistic[0][0])
    p_val = float(wald_test.pvalue)

    print(f"\n  Model A (additive):    R2 = {model_a.rsquared:.4f}")
    print(f"  Model B (interaction): R2 = {model_b.rsquared:.4f}")
    print(f"  delta-R2 = {model_b.rsquared - model_a.rsquared:.4f}")
    print(
        f"  Wald F-test (interactions = 0):"
        f" F = {f_stat:.2f}, p = {p_val:.4f} {_sigstars(p_val)}"
    )

    # Individual interaction coefficients
    print("\n  Interaction coefficients (relative to detached):")
    print(f"  {'Type':<22} {'B':>8} {'SE':>8} {'t':>7} {'p':>8}")
    print("  " + "-" * 58)

    base_slope = model_b.params["log_energy_pc"]
    base_se = model_b.bse["log_energy_pc"]
    base_p = model_b.pvalues["log_energy_pc"]
    print(
        f"  {'Detached (base)':<22}"
        f" {base_slope:>8.4f}"
        f" {base_se:>8.4f}"
        f" {base_slope / base_se:>7.2f}"
        f" {base_p:>8.4f}"
    )

    compounding_found = False
    for param in interaction_params:
        coef = model_b.params[param]
        se = model_b.bse[param]
        t = coef / se if se > 0 else 0
        p = model_b.pvalues[param]

        type_name = param.split("T.")[1].split("]")[0] if "T." in param else param
        label = TYPE_LABELS.get(type_name, type_name)

        print(
            f"  {label:<22} {coef:>8.4f} {se:>8.4f} {t:>7.2f} {p:>8.4f} {_sigstars(p)}"
        )

        # Positive interaction for compact types = more city per energy
        if type_name in ("mid_terrace", "flat") and coef > 0 and p < 0.05:
            compounding_found = True

    # Total slope for each type
    print("\n  Total energy->city slope by type:")
    for t in TYPE_ORDER:
        if t == "detached":
            total_slope = base_slope
        else:
            interaction_key = [p for p in interaction_params if f"T.{t}]" in p]
            if interaction_key:
                total_slope = base_slope + model_b.params[interaction_key[0]]
            else:
                continue
        print(f"    {TYPE_LABELS[t]:<22} {total_slope:>+.4f}")

    print("\n  Interpretation:")
    print("    The base slope is the energy->city elasticity for detached homes.")
    print("    Positive interactions mean that type gets MORE city per unit energy.")

    if compounding_found:
        print("\n  PASS: Compact types have significantly steeper energy->city slopes.")
    else:
        print("\n  FAIL: Interaction terms not significant for compact types.")

    return compounding_found


# ---------------------------------------------------------------------------
# D. Return on Energy (descriptive)
# ---------------------------------------------------------------------------


def report_return_on_energy(df: pd.DataFrame) -> None:
    """
    Report city output per unit energy by morphological type.

    Inverts the framing: not "how much energy does this cost?" but
    "how much city does each kWh buy?" Higher is better.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with ``city_pca`` and ``city_entropy`` columns.
    """
    print("\n" + "=" * 70)
    print("D. RETURN ON ENERGY: How much city does each kWh buy?")
    print("=" * 70)

    has_pca = "city_pca" in df.columns
    has_entropy = "city_entropy" in df.columns

    if not has_pca and not has_entropy:
        print("  SKIP: No city index available")
        return

    # Shift PCA to positive for ratio
    if has_pca:
        min_pca = df["city_pca"].min()
        shift = abs(min_pca) + 1.0 if min_pca <= 0 else 0.0
        df["roe_pca"] = (df["city_pca"] + shift) / df["total_energy_per_capita"].clip(
            lower=1
        )

    if has_entropy:
        df["roe_entropy"] = (df["city_entropy"] + 0.1) / df[
            "total_energy_per_capita"
        ].clip(lower=1)

    header = f"  {'Type':<18} {'N':>8} {'kWh/cap':>9}"
    if has_pca:
        header += f" {'USI':>8} {'ROE_USI':>10}"
    if has_entropy:
        header += f" {'H':>8} {'ROE_H':>10}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    type_roe: dict[str, dict[str, float]] = {}
    for t in TYPE_ORDER:
        sub = df[df["morph_type"] == t]
        if len(sub) < 10:
            continue
        row = (
            f"  {TYPE_LABELS[t]:<18}"
            f" {len(sub):>8,}"
            f" {sub['total_energy_per_capita'].mean():>9,.0f}"
        )
        stats_dict: dict[str, float] = {
            "kwh_cap": sub["total_energy_per_capita"].mean(),
        }

        if has_pca:
            usi = sub["city_pca"].mean()
            roe = sub["roe_pca"].mean()
            row += f" {usi:>8.2f} {roe:>10.6f}"
            stats_dict["roe_pca"] = roe
        if has_entropy:
            h = sub["city_entropy"].mean()
            roe_h = sub["roe_entropy"].mean()
            row += f" {h:>8.3f} {roe_h:>10.6f}"
            stats_dict["roe_entropy"] = roe_h

        type_roe[t] = stats_dict
        print(row)

    if "detached" in type_roe and "flat" in type_roe:
        print("\n  Flat / Detached return-on-energy ratio:")
        if has_pca and type_roe["detached"].get("roe_pca", 0) > 0:
            r = type_roe["flat"]["roe_pca"] / type_roe["detached"]["roe_pca"]
            print(f"    USI (PCA):  {r:.2f}x")
        if has_entropy and type_roe["detached"].get("roe_entropy", 0) > 0:
            r = type_roe["flat"]["roe_entropy"] / type_roe["detached"]["roe_entropy"]
            print(f"    Entropy:    {r:.2f}x")

        print("\n  This is the urban trophic efficiency ratio.")
        print("  A flat in a mixed-use neighbourhood extracts N times more")
        print("  city from each kWh than a detached house in suburbia.")


# ---------------------------------------------------------------------------
# E. Bettencourt Scaling Test
# ---------------------------------------------------------------------------


def test_scaling_residuals(df: pd.DataFrame) -> None:
    """
    Test whether compact OAs produce more GVA than expected from population.

    Bettencourt et al. (2007): GVA ~ N^beta, with beta > 1 (superlinear).
    If compact form is the mechanism, then controlling for morphology
    should absorb some of the superlinear scaling. And compact OAs should
    have positive residuals (more productive than expected).

    Model:
        log(GVA) ~ alpha + beta * log(pop) + gamma * compact_share + epsilon

    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with ``lsoa_gva_millions`` and ``OA21CD`` columns.
    """
    print("\n" + "=" * 70)
    print("E. BETTENCOURT SCALING: Does compact form predict GVA residuals?")
    print("=" * 70)

    gva_col = "lsoa_gva_millions"
    if gva_col not in df.columns or not df[gva_col].notna().any():
        print("  SKIP: No GVA data")
        return

    if "OA21CD" not in df.columns:
        print("  SKIP: No OA codes")
        return

    # Find population column
    pop_col = None
    for col in df.columns:
        if col.startswith("ts001_") and "total" in col.lower():
            pop_col = col
            break
    if pop_col is None:
        print("  SKIP: No Census population column")
        return

    # Aggregate to LSOA level (since GVA is at LSOA)
    lsoa_df = (
        df.groupby("LSOA21CD")
        .agg(
            gva=(gva_col, "first"),
            pop=(pop_col, "sum"),
            total_energy=("total_energy_kwh", "sum"),
            n_dwellings=("morph_type", "count"),
            n_compact=(
                "morph_type",
                lambda x: x.isin(["mid_terrace", "end_terrace", "flat"]).sum(),
            ),
        )
        .reset_index()
    )
    lsoa_df["compact_share"] = lsoa_df["n_compact"] / lsoa_df["n_dwellings"].replace(
        0, np.nan
    )
    lsoa_df = lsoa_df[
        (lsoa_df["gva"] > 0) & (lsoa_df["pop"] > 0) & lsoa_df["compact_share"].notna()
    ].copy()

    lsoa_df["log_gva"] = np.log(lsoa_df["gva"])
    lsoa_df["log_pop"] = np.log(lsoa_df["pop"])
    lsoa_df["energy_per_capita"] = lsoa_df["total_energy"] / lsoa_df["pop"]

    n_lsoa = len(lsoa_df)
    print(f"  LSOAs with GVA + population: {n_lsoa:,}")

    if n_lsoa < 30:
        print("  SKIP: Too few LSOAs")
        return

    # Model 1: pure scaling law
    m1 = smf.ols("log_gva ~ log_pop", data=lsoa_df).fit(cov_type="HC3")
    beta = m1.params["log_pop"]
    beta_p = m1.pvalues["log_pop"]
    print(
        f"\n  Pure scaling: log(GVA) = {m1.params['Intercept']:.3f}"
        f" + {beta:.3f} * log(pop)"
    )
    print(f"  beta = {beta:.3f} {_sigstars(beta_p)} (1.0 = linear, >1 = superlinear)")
    print(f"  R2 = {m1.rsquared:.4f}")

    # Model 2: + compact_share
    m2 = smf.ols("log_gva ~ log_pop + compact_share", data=lsoa_df).fit(cov_type="HC3")
    gamma = m2.params["compact_share"]
    gamma_p = m2.pvalues["compact_share"]
    print(f"\n  + compact_share: gamma = {gamma:.4f} {_sigstars(gamma_p)}")
    print(f"  R2 = {m2.rsquared:.4f} (delta = {m2.rsquared - m1.rsquared:+.4f})")

    if gamma > 0 and gamma_p < 0.05:
        print("\n  Compact morphology predicts HIGHER GVA than expected from")
        print("  population alone. The urban conduit amplifies output.")
    elif gamma > 0:
        print("\n  Positive but not significant. Suggestive only.")
    else:
        print("\n  Compact share does not predict GVA residuals.")

    # Residuals by compact share quintile
    lsoa_df["scaling_residual"] = m1.resid
    lsoa_df["compact_q"] = pd.qcut(
        lsoa_df["compact_share"],
        q=5,
        labels=["Q1 sprawl", "Q2", "Q3", "Q4", "Q5 compact"],
    )

    print("\n  Scaling residuals by compact share quintile:")
    print(
        f"  {'Quintile':<14} {'N':>5} {'Compact%':>9} {'Mean resid':>11} {'kWh/cap':>9}"
    )
    print("  " + "-" * 52)
    for q in ["Q1 sprawl", "Q2", "Q3", "Q4", "Q5 compact"]:
        sub = lsoa_df[lsoa_df["compact_q"] == q]
        if len(sub) == 0:
            continue
        print(
            f"  {q:<14}"
            f" {len(sub):>5}"
            f" {sub['compact_share'].mean() * 100:>8.1f}%"
            f" {sub['scaling_residual'].mean():>+11.4f}"
            f" {sub['energy_per_capita'].mean():>9,.0f}"
        )

    # Correlation between scaling residual and energy per capita
    r, p = sp_stats.pearsonr(lsoa_df["scaling_residual"], lsoa_df["energy_per_capita"])
    print(
        f"\n  Correlation: scaling residual vs energy/capita:"
        f" r = {r:.3f} {_sigstars(p)}"
    )
    if r < 0 and p < 0.05:
        print("  Areas that produce MORE than expected also consume LESS per capita.")
        print(
            "  This is the dual scaling: compact form saves energy"
            " AND amplifies output."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_analysis(df: pd.DataFrame, label: str = "") -> tuple[bool, bool]:
    """
    Run the full trophic framework analysis on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data (from ``load_data``).
    label : str
        Label for output headers (e.g. city name).

    Returns
    -------
    tuple[bool, bool]
        (bootstrap_passed, interaction_passed).
    """
    if label:
        print("\n" + "#" * 70)
        print(f"# {label.upper()}")
        print("#" * 70)

    df = construct_city_indices(df)
    gate_b = test_denominator_progression(df)
    gate_c = test_interaction_model(df)
    report_return_on_energy(df)
    test_scaling_residuals(df)

    print("\n" + "=" * 70)
    print(f"SUMMARY{f' — {label}' if label else ''}")
    print("=" * 70)
    print(f"  Bootstrap progression:  {'PASS' if gate_b else 'FAIL'}")
    print(f"  Interaction model:      {'PASS' if gate_c else 'FAIL'}")

    if gate_b and gate_c:
        print("\n  Both tests confirm: compact form moderates the")
        print("  energy->city relationship. The compounding is real.")
    elif gate_b or gate_c:
        print("\n  Mixed results. One test supports compounding,")
        print("  the other does not. Investigate further.")
    else:
        print("\n  Neither test supports the compounding claim.")
        print("  The efficiency gap may be entirely explained by")
        print("  building physics and household size.")

    return gate_b, gate_c


def main(scale: str = "uprn") -> None:
    """
    Run the trophic framework per-city and combined.

    Parameters
    ----------
    scale : str
        ``"uprn"`` for dwelling-level (original), ``"oa"`` to aggregate
        to Output Area before analysis (eliminates pseudo-replication).
    """
    import sys as _sys

    print("=" * 70)
    print("TROPHIC FRAMEWORK: Rigorous formalisation")
    print("=" * 70)
    print("  Three tests of the claim that compact morphologies deliver")
    print("  more city per unit of energy consumed.")
    print(f"  Scale: {scale}\n")

    # Load full dataset
    df = load_data()

    # Aggregate to OA if requested
    if scale == "oa":
        df = aggregate_to_oa(df)

    # Detect available cities
    cities = sorted(df["city"].unique()) if "city" in df.columns else []

    # If CLI args given, filter to those cities (exclude "oa"/"uprn")
    args = [a for a in _sys.argv[1:] if a not in ("oa", "uprn")]
    if args and cities:
        cities = [c for c in cities if c in args]
        if not cities:
            print(f"  Unknown cities: {args}")
            print(f"  Available: {sorted(df['city'].unique())}")
            _sys.exit(1)

    results: dict[str, tuple[bool, bool]] = {}

    # Per-city analysis
    if len(cities) > 1:
        for city in cities:
            city_df = df[df["city"] == city].copy()
            n = len(city_df)
            min_n = 100 if scale == "oa" else 500
            if n < min_n:
                print(f"\n  SKIP {city}: only {n} records (need >= {min_n})")
                continue
            results[city] = run_analysis(city_df, label=city)

    # Combined analysis (always)
    label = "ALL CITIES COMBINED" if cities else ""
    results["combined"] = run_analysis(df, label=label)

    # Cross-city comparison table
    if len(results) > 2:
        print("\n" + "=" * 70)
        print("CROSS-CITY COMPARISON")
        print("=" * 70)
        print(f"  {'City':<20} {'N':>8} {'Bootstrap':>12} {'Interaction':>12}")
        print("  " + "-" * 55)
        for city in cities:
            if city not in results:
                continue
            n = len(df[df["city"] == city])
            b, c = results[city]
            print(
                f"  {city:<20} {n:>8,}"
                f" {'PASS' if b else 'FAIL':>12}"
                f" {'PASS' if c else 'FAIL':>12}"
            )
        b, c = results["combined"]
        print(
            f"  {'COMBINED':<20} {len(df):>8,}"
            f" {'PASS' if b else 'FAIL':>12}"
            f" {'PASS' if c else 'FAIL':>12}"
        )


if __name__ == "__main__":
    _scale = "oa" if "oa" in sys.argv[1:] else "uprn"
    main(scale=_scale)
