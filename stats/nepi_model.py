"""
NEPI Planning Tool: XGBoost models with SHAP explanations.

Trains two gradient-boosted models to predict neighbourhood energy costs
from four planner-controllable inputs:

    1. Population density (people/ha)
    2. Built form (detached ↔ flat mix)
    3. Amenity access (walkable service coverage, 0–1)
    4. Transit access (bus + rail gravity-weighted count)

Model A predicts Form (building energy, kWh/hh/yr).
Model B predicts Mobility (transport energy, kWh/hh/yr).
Total = Form + Mobility, banded A–G by national percentile.

SHAP TreeExplainer provides exact feature attributions for each prediction,
showing planners which inputs drive energy costs up or down.

Usage:
    uv run python stats/nepi_model.py          # train and save models
    uv run streamlit run stats/nepi_app.py     # launch planning tool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from nepi import BAND_COLORS, BAND_PERCENTILES, compute_local_coverage  # noqa: E402
from proof_of_concept_oa import build_accessibility, load_and_aggregate  # noqa: E402
from urban_energy.paths import DATA_DIR  # noqa: E402

MODEL_DIR = DATA_DIR / "models" / "nepi"
FIGURE_DIR = Path(__file__).parent / "figures" / "nepi"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# The 9 features fed to XGBoost (planner inputs only — no mediators)
FEATURES: list[str] = [
    "people_per_ha",       # Slider 1: Density
    "pct_detached",        # Slider 2: Built form
    "pct_semi",            # Slider 2: Built form
    "pct_terraced",        # Slider 2: Built form
    "pct_flat",            # Slider 2: Built form
    "local_coverage",      # Slider 3: Amenity access
    "cc_bus_800_wt",       # Slider 4: Transit access
    "cc_rail_800_wt",      # Slider 4: Transit access
    "median_build_year",   # Slider 5: Building era
]

# Human-readable labels for SHAP plots
FEATURE_LABELS: dict[str, str] = {
    "people_per_ha": "Population density",
    "pct_detached": "% Detached",
    "pct_semi": "% Semi-detached",
    "pct_terraced": "% Terraced",
    "pct_flat": "% Flat",
    "local_coverage": "Amenity access",
    "cc_bus_800_wt": "Bus access",
    "cc_rail_800_wt": "Rail access",
    "median_build_year": "Building era",
}

# Targets: 2 energy surfaces + 2 behavioural outcomes
TARGET_FORM = "building_kwh_per_hh"
TARGET_MOBILITY = "transport_kwh_per_hh_total_est"
TARGET_CARS = "cars_per_hh"
TARGET_COMMUTE = "avg_commute_km"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_training_data() -> tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
]:
    """
    Load OA data and extract features and targets for model training.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (N × 9).
    y_form : pd.Series
        Building energy target (kWh/hh/yr).
    y_mobility : pd.Series
        Transport energy target (kWh/hh/yr).
    y_cars : pd.Series
        Cars per household target.
    y_commute : pd.Series
        Average commute distance target (km).
    dominant_type : pd.Series
        Housing type for stratified splitting.
    """
    lsoa = load_and_aggregate()
    lsoa = build_accessibility(lsoa)
    lsoa = compute_local_coverage(lsoa)

    # Keep only rows with all features and all targets
    all_targets = [TARGET_FORM, TARGET_MOBILITY, TARGET_CARS, TARGET_COMMUTE]
    cols_needed = FEATURES + all_targets + ["dominant_type"]
    df = lsoa[cols_needed].dropna().copy()

    print(f"\n  Training data: {len(df):,} OAs with complete features and targets")

    X = df[FEATURES]
    y_form = df[TARGET_FORM]
    y_mobility = df[TARGET_MOBILITY]
    y_cars = df[TARGET_CARS]
    y_commute = df[TARGET_COMMUTE]
    dominant_type = df["dominant_type"]

    return X, y_form, y_mobility, y_cars, y_commute, dominant_type


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


# Monotonic constraints per feature per target.
# +1 = feature increase → prediction must increase (or stay flat)
# -1 = feature increase → prediction must decrease (or stay flat)
#  0 = unconstrained
#
# Feature order: people_per_ha, pct_detached, pct_semi, pct_terraced,
#   pct_flat, local_coverage, cc_bus_800_wt, cc_rail_800_wt, median_build_year
# Per-model feature subsets — each model only sees causally relevant features.
# This prevents cross-contamination (e.g., transit affecting building energy).
MODEL_FEATURES: dict[str, list[str]] = {
    "form": [
        "people_per_ha", "pct_detached", "pct_semi", "pct_terraced",
        "pct_flat", "median_build_year",
    ],
    "mobility": [
        "people_per_ha", "pct_detached", "pct_semi", "pct_terraced",
        "pct_flat", "local_coverage", "cc_bus_800_wt", "cc_rail_800_wt",
    ],
    "cars": [
        "people_per_ha", "pct_detached", "pct_semi", "pct_terraced",
        "pct_flat", "local_coverage", "cc_bus_800_wt", "cc_rail_800_wt",
    ],
    "commute": [
        "local_coverage", "cc_bus_800_wt", "cc_rail_800_wt",
    ],
}

# Monotonic constraints per model (same order as MODEL_FEATURES[model]).
# +1 = feature increase → prediction must increase
# -1 = feature increase → prediction must decrease
#  0 = unconstrained
MONOTONIC_CONSTRAINTS: dict[str, tuple[int, ...]] = {
    # Form: denser/flatter/newer = less building energy
    "form": (-1, +1, 0, 0, -1, -1),
    # Mobility: denser/flatter/more access/more transit = less transport energy
    "mobility": (-1, +1, 0, 0, -1, -1, -1, -1),
    # Cars: denser/flatter/more access/more transit = fewer cars
    "cars": (-1, +1, 0, 0, -1, -1, -1, -1),
    # Commute: more access/more transit = shorter commute
    "commute": (-1, -1, -1),
}


def _train_one_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
    monotone: tuple[int, ...] | None = None,
    feature_subset: list[str] | None = None,
) -> xgb.XGBRegressor:
    """
    Train a single XGBoost regressor with early stopping.

    Parameters
    ----------
    label : str
        Model name for printing.
    monotone : tuple of int or None
        Per-feature monotonic constraints (+1, -1, 0).

    Returns
    -------
    xgb.XGBRegressor
        Trained model.
    """
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        base_score=0.0,  # explicit to avoid SHAP/XGBoost JSON parsing issue
        monotone_constraints=monotone,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    # Subset features if specified
    if feature_subset:
        X_train = X_train[feature_subset]
        X_test = X_test[feature_subset]

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  {label} model:")
    print(f"    Best iteration: {model.best_iteration}")
    print(f"    Test MAE:  {mae:,.0f} kWh")
    print(f"    Test RMSE: {rmse:,.0f} kWh")
    print(f"    Test R²:   {r2:.3f}")

    return model


def train_models(
    X: pd.DataFrame,
    y_form: pd.Series,
    y_mobility: pd.Series,
    y_cars: pd.Series,
    y_commute: pd.Series,
    dominant_type: pd.Series,
) -> dict[str, xgb.XGBRegressor]:
    """
    Train 4 models with stratified train/test split.

    Returns
    -------
    dict mapping model name to trained XGBRegressor.
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODELS (4 targets)")
    print("=" * 60)

    # Stratified split by dominant housing type
    split = train_test_split(
        X, y_form, y_mobility, y_cars, y_commute, dominant_type,
        test_size=0.2, random_state=42, stratify=dominant_type,
    )
    X_train, X_test = split[0], split[1]
    yf_train, yf_test = split[2], split[3]
    ym_train, ym_test = split[4], split[5]
    yc_train, yc_test = split[6], split[7]
    yk_train, yk_test = split[8], split[9]
    dt_train, dt_test = split[10], split[11]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    targets = [
        ("form", yf_train, yf_test, "Form (kWh/hh)"),
        ("mobility", ym_train, ym_test, "Mobility (kWh/hh)"),
        ("cars", yc_train, yc_test, "Cars per household"),
        ("commute", yk_train, yk_test, "Commute distance (km)"),
    ]

    trained: dict[str, xgb.XGBRegressor] = {}
    test_data: dict[str, pd.Series] = {}
    for name, y_tr, y_te, label in targets:
        monotone = MONOTONIC_CONSTRAINTS.get(name)
        feat_subset = MODEL_FEATURES.get(name)
        trained[name] = _train_one_model(
            X_train, y_tr, X_test, y_te, label,
            monotone=monotone, feature_subset=feat_subset,
        )
        test_data[name] = y_te

    # Per-type evaluation
    print("\n  Per-type test R²:")
    for dtype in ["Flat", "Terraced", "Semi", "Detached"]:
        mask = dt_test == dtype
        if mask.sum() < 10:
            continue
        for name, y_te, label in [
            ("form", yf_test, "Form"),
            ("mobility", ym_test, "Mobility"),
            ("cars", yc_test, "Cars"),
            ("commute", yk_test, "Commute"),
        ]:
            feat_cols = MODEL_FEATURES.get(name, FEATURES)
            y_t = y_te[mask]
            y_p = trained[name].predict(X_test.loc[mask, feat_cols])
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - y_t.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            print(f"    {dtype:<12s} {label:<10s} R²={r2:.3f} (N={mask.sum():,})")

    return trained


# ---------------------------------------------------------------------------
# NEPI band computation
# ---------------------------------------------------------------------------


def compute_band_thresholds(y_form: pd.Series, y_mobility: pd.Series) -> dict:
    """
    Compute absolute kWh thresholds for NEPI bands from the national distribution.

    Parameters
    ----------
    y_form, y_mobility : pd.Series
        Full training data (not just test set).

    Returns
    -------
    dict
        Mapping band letter -> (lower_kwh, upper_kwh).
    """
    total = y_form + y_mobility
    thresholds = {}
    for band, (plo, phi) in BAND_PERCENTILES.items():
        lo = float(np.percentile(total, plo))
        hi = float(np.percentile(total, phi))
        thresholds[band] = {"lo": round(lo, 0), "hi": round(hi, 0)}
    return thresholds


def assign_band(total_kwh: float, thresholds: dict) -> str:
    """Assign A–G band from total kWh/hh/yr."""
    for band in ["A", "B", "C", "D", "E", "F", "G"]:
        if total_kwh <= thresholds[band]["hi"]:
            return band
    return "G"


# ---------------------------------------------------------------------------
# Archetype profiles and feature stats
# ---------------------------------------------------------------------------


def compute_archetypes(X: pd.DataFrame, dominant_type: pd.Series) -> dict:
    """
    Compute median feature vectors for each dominant housing type.

    Used for preset buttons and built-form slider interpolation.
    """
    archetypes = {}
    for dtype in ["Flat", "Terraced", "Semi", "Detached"]:
        mask = dominant_type == dtype
        if mask.sum() < 10:
            continue
        medians = X[mask].median()
        archetypes[dtype] = {col: round(float(medians[col]), 2) for col in FEATURES}
    return archetypes


def compute_feature_stats(X: pd.DataFrame) -> dict:
    """Compute P5/P50/P95 for each feature (slider calibration + OOD check)."""
    stats = {}
    for col in FEATURES:
        vals = X[col].dropna()
        stats[col] = {
            "p5": round(float(vals.quantile(0.05)), 2),
            "p50": round(float(vals.median()), 2),
            "p95": round(float(vals.quantile(0.95)), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
        }
    return stats


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def _patch_shap_xgboost_compat() -> None:
    """
    Monkey-patch shap to handle xgboost>=2.0 base_score JSON array format.

    XGBoost 2.0+ stores base_score as '[0E0]' in JSON config; shap<0.50
    expects a plain float string like '0.5'. This patches the specific
    float() call in XGBTreeModelLoader.__init__ to strip array brackets.
    """
    import builtins

    loader_cls = shap.explainers._tree.XGBTreeModelLoader
    original_init = loader_cls.__init__

    def patched_init(self: object, xgb_model: object) -> None:
        _original_float = builtins.float

        def _safe_float(val: object) -> float:
            if isinstance(val, str) and val.startswith("["):
                val = val.strip("[]")
            return _original_float(val)

        builtins.float = _safe_float  # type: ignore[assignment]
        try:
            original_init(self, xgb_model)
        finally:
            builtins.float = _original_float

    loader_cls.__init__ = patched_init  # type: ignore[method-assign]


_patch_shap_xgboost_compat()


def _make_explainer(
    model: xgb.XGBRegressor,
    background: pd.DataFrame,
) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for an XGBoost model."""
    return shap.TreeExplainer(model)


def make_shap_waterfall(
    explainer: shap.Explainer,
    features: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer for the model.
    features : pd.DataFrame
        Single-row DataFrame with the 8 features.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sv = explainer(features)
    # Label each feature with its value for clarity
    labels = []
    for f in features.columns:
        name = FEATURE_LABELS.get(f, f)
        val = features[f].iloc[0]
        if "pct_" in f:
            labels.append(f"{name} = {val:.0f}%")
        elif f == "local_coverage":
            labels.append(f"{name} = {val:.0%}")
        elif f == "median_build_year":
            labels.append(f"{name} = {val:.0f}")
        elif "800_wt" in f:
            labels.append(f"{name} = {val:.1f}")
        else:
            labels.append(f"{name} = {val:.0f}")
    sv.feature_names = labels

    shap.plots.waterfall(sv[0], max_display=8, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02, color="#cccccc")
    fig.set_facecolor("none")
    fig.set_size_inches(7, 3.5)  # fixed size to prevent reflow
    for ax in fig.get_axes():
        ax.set_facecolor("none")
        ax.tick_params(labelsize=9, colors="#999999")
        for spine in ax.spines.values():
            spine.set_color("#444444")
    fig.set_dpi(200)
    plt.tight_layout()
    return fig


def save_global_shap(
    model_form: xgb.XGBRegressor,
    model_mobility: xgb.XGBRegressor,
    X_sample: pd.DataFrame,
) -> None:
    """Save global SHAP summary bar plots for both models."""
    for model, label, name in [
        (model_form, "Form", "form"),
        (model_mobility, "Mobility", "mobility"),
    ]:
        feat_cols = MODEL_FEATURES[name]
        explainer = _make_explainer(model, X_sample[feat_cols])
        sv = explainer(X_sample[feat_cols])
        sv.feature_names = [FEATURE_LABELS.get(f, f) for f in feat_cols]

        shap.plots.bar(sv, max_display=8, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Global Feature Importance: {label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / f"nepi_shap_global_{label.lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved nepi_shap_global_{label.lower()}.png")


# ---------------------------------------------------------------------------
# Prediction API (used by Streamlit app)
# ---------------------------------------------------------------------------


def interpolate_archetype(
    position: float,
    archetypes: dict,
) -> dict[str, float]:
    """
    Interpolate all features along the archetype gradient.

    Parameters
    ----------
    position : float
        Morphology slider (0 = detached, 1 = flat).
    archetypes : dict
        National archetype profiles per dominant type.

    Returns
    -------
    dict mapping feature name to interpolated value.
    """
    type_positions = {"Detached": 0.0, "Semi": 0.33, "Terraced": 0.67, "Flat": 1.0}

    # Gaussian kernel weights — smooth blending between archetypes
    weights: dict[str, float] = {}
    for dtype, pos in type_positions.items():
        if dtype not in archetypes:
            weights[dtype] = 0.0
            continue
        weights[dtype] = np.exp(-0.5 * ((position - pos) / 0.2) ** 2)

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0

    # Weighted average of ALL features across archetypes
    result: dict[str, float] = {}
    for col in FEATURES:
        result[col] = sum(
            weights[dtype] / total_weight * archetypes[dtype].get(col, 0.0)
            for dtype in weights
            if dtype in archetypes
        )

    # Normalise housing percentages to sum to 100%
    pct_cols = ["pct_detached", "pct_semi", "pct_terraced", "pct_flat"]
    pct_sum = sum(result[c] for c in pct_cols)
    if pct_sum > 0:
        for c in pct_cols:
            result[c] = result[c] / pct_sum * 100

    return result


def build_feature_vector(
    archetype_values: dict[str, float],
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a feature vector from archetype baseline + optional overrides.

    Parameters
    ----------
    archetype_values : dict
        Interpolated archetype values for all features.
    overrides : dict or None
        Feature-level overrides. Only specified features are replaced.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with model features.
    """
    row = dict(archetype_values)
    if overrides:
        for k, v in overrides.items():
            if k in row:
                row[k] = v
    return pd.DataFrame([row])[FEATURES]


def expand_sliders(
    density: float,
    form_position: float,
    amenity_access: float,
    transit_access: float,
    archetypes: dict,
    build_year: float = 1970,
) -> pd.DataFrame:
    """
    Expand slider values to feature DataFrame for model input.

    Convenience wrapper for backward compatibility with test scenarios.
    """
    arch = interpolate_archetype(form_position, archetypes)
    overrides = {
        "people_per_ha": density,
        "local_coverage": amenity_access,
        "cc_bus_800_wt": transit_access * 0.86,
        "cc_rail_800_wt": transit_access * 0.14,
        "median_build_year": build_year,
    }
    return build_feature_vector(arch, overrides)


def load_models() -> dict:
    """
    Load trained models and metadata from disk.

    Returns
    -------
    dict with keys: model_form, model_mobility, explainer_form,
    explainer_mobility, band_thresholds, archetypes, feature_stats
    """
    loaded: dict[str, xgb.XGBRegressor] = {}
    for name in ["form", "mobility", "cars", "commute"]:
        m = xgb.XGBRegressor()
        m.load_model(MODEL_DIR / f"nepi_model_{name}.json")
        loaded[name] = m

    with open(MODEL_DIR / "nepi_band_thresholds.json") as f:
        band_thresholds = json.load(f)
    with open(MODEL_DIR / "nepi_archetype_profiles.json") as f:
        archetypes = json.load(f)
    with open(MODEL_DIR / "nepi_feature_stats.json") as f:
        feature_stats = json.load(f)

    return {
        "model_form": loaded["form"],
        "model_mobility": loaded["mobility"],
        "model_cars": loaded["cars"],
        "model_commute": loaded["commute"],
        "explainer_form": _make_explainer(loaded["form"], pd.DataFrame()),
        "explainer_mobility": _make_explainer(loaded["mobility"], pd.DataFrame()),
        "band_thresholds": band_thresholds,
        "archetypes": archetypes,
        "feature_stats": feature_stats,
    }


def predict_nepi(
    density: float,
    form_position: float,
    amenity_access: float,
    transit_access: float,
    models: dict,
    build_year: float = 1970,
) -> dict:
    """
    Predict NEPI from slider values.

    Parameters
    ----------
    density : float
        Population density (people/ha).
    form_position : float
        Built form (0=detached, 1=flat).
    amenity_access : float
        Walkable service coverage (0–1).
    transit_access : float
        Transit gravity-weighted count.
    models : dict
        Output of load_models().
    build_year : float
        Median building construction year.

    Returns
    -------
    dict with form_kwh, mobility_kwh, total_kwh, band, cars_per_hh,
    avg_commute_km, features, shap_form, shap_mobility
    """
    arch = interpolate_archetype(form_position, models["archetypes"])
    overrides = {}
    if density is not None:
        overrides["people_per_ha"] = density
    if amenity_access is not None:
        overrides["local_coverage"] = amenity_access
    if transit_access is not None:
        overrides["cc_bus_800_wt"] = transit_access * 0.86
        overrides["cc_rail_800_wt"] = transit_access * 0.14
    if build_year is not None:
        overrides["median_build_year"] = build_year
    features = build_feature_vector(arch, overrides if overrides else None)

    form_kwh = float(models["model_form"].predict(features[MODEL_FEATURES["form"]])[0])
    mobility_kwh = float(models["model_mobility"].predict(features[MODEL_FEATURES["mobility"]])[0])
    cars = float(models["model_cars"].predict(features[MODEL_FEATURES["cars"]])[0])
    commute = float(models["model_commute"].predict(features[MODEL_FEATURES["commute"]])[0])
    total_kwh = form_kwh + mobility_kwh
    band = assign_band(total_kwh, models["band_thresholds"])

    # SHAP values (use model-specific feature subsets)
    sv_form = models["explainer_form"](features[MODEL_FEATURES["form"]])
    sv_mobility = models["explainer_mobility"](features[MODEL_FEATURES["mobility"]])

    return {
        "form_kwh": form_kwh,
        "mobility_kwh": mobility_kwh,
        "total_kwh": total_kwh,
        "band": band,
        "cars_per_hh": cars,
        "avg_commute_km": commute,
        "amenity_access": amenity_access,
        "features": features,
        "shap_form": sv_form,
        "shap_mobility": sv_mobility,
    }


# ---------------------------------------------------------------------------
# Main: train and save
# ---------------------------------------------------------------------------


def main() -> None:
    """Train models and save all artifacts."""
    print("=" * 60)
    print("NEPI PLANNING TOOL: MODEL TRAINING")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare data
    X, y_form, y_mobility, y_cars, y_commute, dominant_type = prepare_training_data()

    # Train
    trained = train_models(X, y_form, y_mobility, y_cars, y_commute, dominant_type)

    # Save models
    for name, model in trained.items():
        model.save_model(MODEL_DIR / f"nepi_model_{name}.json")
    print(f"\n  Models saved to {MODEL_DIR}")

    # Band thresholds
    thresholds = compute_band_thresholds(y_form, y_mobility)
    with open(MODEL_DIR / "nepi_band_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("  Band thresholds:")
    for band, t in thresholds.items():
        print(f"    {band}: {t['lo']:,.0f} – {t['hi']:,.0f} kWh/hh/yr")

    # Archetypes
    archetypes = compute_archetypes(X, dominant_type)
    with open(MODEL_DIR / "nepi_archetype_profiles.json", "w") as f:
        json.dump(archetypes, f, indent=2)
    print("  Archetype profiles saved")

    # Feature stats
    stats = compute_feature_stats(X)
    with open(MODEL_DIR / "nepi_feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  Feature statistics saved")

    # Global SHAP summary (use sample for speed)
    sample_n = min(5000, len(X))
    X_sample = X.sample(n=sample_n, random_state=42)
    save_global_shap(trained["form"], trained["mobility"], X_sample)

    # Quick prediction test
    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)

    models = {
        "model_form": trained["form"],
        "model_mobility": trained["mobility"],
        "model_cars": trained["cars"],
        "model_commute": trained["commute"],
        "explainer_form": _make_explainer(trained["form"], pd.DataFrame()),
        "explainer_mobility": _make_explainer(trained["mobility"], pd.DataFrame()),
        "band_thresholds": thresholds,
        "archetypes": archetypes,
        "feature_stats": stats,
    }

    # Scenarios: density, form, amenity, transit, year
    scenarios = [
        ("Urban flat (new)",      80, 0.95, 0.85, 25, 2010),
        ("Urban flat (old)",      80, 0.95, 0.85, 25, 1960),
        ("Suburban semi",         35, 0.40, 0.55,  8, 1975),
        ("Rural detached (new)",  10, 0.05, 0.30,  2, 2005),
        ("Rural detached (old)",  10, 0.05, 0.30,  2, 1950),
        ("Ideal compact (new)",   60, 0.90, 0.80, 20, 2015),
    ]

    print(
        f"\n  {'Scenario':<25s} {'Form':>7s} {'Mobil':>7s} {'Total':>7s} "
        f"{'Band':>5s} {'Cars':>5s} {'km':>5s}"
    )
    print(f"  {'-' * 65}")
    for name, density, form_pos, amenity, transit, yr in scenarios:
        result = predict_nepi(
            density, form_pos, amenity, transit, models, build_year=yr,
        )
        print(
            f"  {name:<25s} {result['form_kwh']:>7,.0f} "
            f"{result['mobility_kwh']:>7,.0f} {result['total_kwh']:>7,.0f} "
            f"{result['band']:>5s} {result['cars_per_hh']:>5.1f} "
            f"{result['avg_commute_km']:>5.1f}"
        )

    print(f"\n{'=' * 60}")
    print("DONE — run `uv run streamlit run stats/nepi_app.py` to launch")
    print("=" * 60)


if __name__ == "__main__":
    main()
