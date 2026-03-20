"""
NEPI Planning Tool: Interactive Streamlit Application.

One primary morphology slider drives all features along the observed
archetype gradient. Individual override sliders allow targeted adjustments.
Four XGBoost models predict energy costs and transport behaviour.
SHAP waterfall plots explain each prediction.

Usage:
    uv run streamlit run stats/nepi_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Light, legible chart style — transparent backgrounds, light text/spines
matplotlib.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "figure.dpi": 200,
    "text.color": "#aaaaaa",
    "axes.labelcolor": "#aaaaaa",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.3,
    "legend.edgecolor": "#444444",
    "font.size": 10,
})

sys.path.insert(0, str(Path(__file__).parent))

from nepi import BAND_COLORS  # noqa: E402
from nepi_model import (  # noqa: E402
    FEATURE_LABELS,
    FEATURES,
    MODEL_FEATURES,
    assign_band,
    build_feature_vector,
    interpolate_archetype,
    load_models,
    make_shap_waterfall,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NEPI Planning Tool",
    page_icon="🏘️",
    layout="wide",
)

# Compact spacing
st.markdown("""<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0.5rem; }
    h1 { margin-bottom: 0.2rem !important; font-size: 1.8rem !important; }
    h3 { margin-top: 0.5rem !important; margin-bottom: 0.3rem !important; }
    .stMetric { padding: 0.2rem 0 !important; }
    .stMetric label { font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    /* Fixed-height chart containers to prevent reflow */
    [data-testid="stImage"], .stPlotlyChart, [data-testid="stVegaLiteChart"] {
        min-height: 300px;
    }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load models (cached across sessions)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_models() -> dict:
    """Load trained models and metadata (cached)."""
    return load_models()


models = get_models()
archetypes = models["archetypes"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Neighbourhood Type")

    # PRIMARY SLIDER
    st.markdown("**Sprawl** ← → **Compact**")

    def _on_morph_change():
        new_arch = interpolate_archetype(st.session_state["morph"], archetypes)
        tc = new_arch["cc_bus_800_wt"] + new_arch["cc_rail_800_wt"]
        st.session_state["coverage"] = round(new_arch["local_coverage"] * 100)
        st.session_state["transit"] = round(tc, 1)
        st.session_state["build_year"] = round(new_arch["median_build_year"])

    morph = st.slider(
        "Neighbourhood morphology",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("morph", 0.5),
        step=0.01, key="morph",
        on_change=_on_morph_change,
        help="Moves all parameters along the observed national gradient.",
        label_visibility="collapsed",
    )

    if morph >= 0.83:
        morph_label = "Compact (flat-dominant)"
    elif morph >= 0.50:
        morph_label = "Moderate-compact (terraced)"
    elif morph >= 0.17:
        morph_label = "Moderate-sprawl (semi-detached)"
    else:
        morph_label = "Sprawl (detached)"
    st.caption(f"**{morph_label}**")

    arch = interpolate_archetype(morph, archetypes)
    transit_combined = arch["cc_bus_800_wt"] + arch["cc_rail_800_wt"]

    st.divider()
    st.subheader("Individual parameters")
    st.caption("Track morphology by default. Adjust for targeted interventions.")

    density_override = arch["people_per_ha"]

    coverage_override = st.slider(
        "Amenity access (%)", min_value=0, max_value=100,
        value=round(arch["local_coverage"] * 100), step=1, key="coverage",
    )
    transit_override = st.slider(
        "Transit access", min_value=0.0, max_value=40.0,
        value=round(transit_combined, 1), step=0.5, key="transit",
    )
    build_year_override = st.slider(
        "Building era", min_value=1900, max_value=2025,
        value=round(arch["median_build_year"]), step=5, key="build_year",
    )

    def _reset_to_archetype():
        a = interpolate_archetype(st.session_state["morph"], archetypes)
        tc = a["cc_bus_800_wt"] + a["cc_rail_800_wt"]
        st.session_state["coverage"] = round(a["local_coverage"] * 100)
        st.session_state["transit"] = round(tc, 1)
        st.session_state["build_year"] = round(a["median_build_year"])

    st.button("Reset to archetype", use_container_width=True, on_click=_reset_to_archetype)

# ---------------------------------------------------------------------------
# Build feature vector
# ---------------------------------------------------------------------------

overrides: dict[str, float] = {}
if abs(density_override - arch["people_per_ha"]) > 1.0:
    overrides["people_per_ha"] = density_override
if abs(coverage_override / 100.0 - arch["local_coverage"]) > 0.01:
    overrides["local_coverage"] = coverage_override / 100.0
if abs(transit_override - transit_combined) > 0.3:
    overrides["cc_bus_800_wt"] = transit_override * 0.86
    overrides["cc_rail_800_wt"] = transit_override * 0.14
if abs(build_year_override - arch["median_build_year"]) > 3:
    overrides["median_build_year"] = float(build_year_override)

features = build_feature_vector(arch, overrides if overrides else None)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

form_kwh = float(models["model_form"].predict(features[MODEL_FEATURES["form"]])[0])
mobility_kwh = float(models["model_mobility"].predict(features[MODEL_FEATURES["mobility"]])[0])
cars = float(models["model_cars"].predict(features[MODEL_FEATURES["cars"]])[0])
commute = float(models["model_commute"].predict(features[MODEL_FEATURES["commute"]])[0])
total_kwh = form_kwh + mobility_kwh
band = assign_band(total_kwh, models["band_thresholds"])

sv_form = models["explainer_form"](features[MODEL_FEATURES["form"]])
sv_mobility = models["explainer_mobility"](features[MODEL_FEATURES["mobility"]])

# ---------------------------------------------------------------------------
# Main panel — compact single-page layout
# ---------------------------------------------------------------------------

st.title("NEPI Planning Tool")

st.warning(
    "**Proof of concept — experimental.** "
    "Predictions are indicative of direction and magnitude, not literal targets. "
    "Models use monotonic constraints to enforce theoretically-grounded directionality "
    "and prevent ecological confounds from producing counterintuitive results.",
    icon="⚠️",
)

# --- Row 1: Band badge + metrics + interpretation (all on one line) ---
col_badge, col_metrics = st.columns([1, 4])

band_color = BAND_COLORS.get(band, "#888888")
with col_badge:
    st.markdown(
        f"""<div style="
            background-color: {band_color}; color: white;
            font-size: 56px; font-weight: bold; text-align: center;
            border-radius: 10px; padding: 12px; width: 100px;
            margin: 0 auto; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        ">{band}</div>
        <p style="text-align:center; font-size:12px; color:#888; margin-top:4px;">NEPI Band</p>""",
        unsafe_allow_html=True,
    )

with col_metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Form", f"{form_kwh:,.0f} kWh")
    m2.metric("Mobility", f"{mobility_kwh:,.0f} kWh")
    m3.metric("Total", f"{total_kwh:,.0f} kWh")
    m4.metric("Cars/hh", f"{cars:.2f}")
    m5.metric("Commute", f"{commute:.1f} km")

# Interpretation
form_share = form_kwh / total_kwh * 100
shap_vals = sv_mobility.values[0]
mob_feat_names = [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES["mobility"]]
top_idx = int(np.argmax(np.abs(shap_vals)))
top_dir = "increasing" if shap_vals[top_idx] > 0 else "reducing"
st.caption(
    f"**Band {band}** — {total_kwh:,.0f} kWh/hh/yr "
    f"(Form {form_share:.0f}%, Mobility {100 - form_share:.0f}%). "
    f"Predicted {cars:.1f} cars/hh, {commute:.1f} km commute. "
    f"Strongest transport driver: **{mob_feat_names[top_idx]}** ({top_dir} costs)."
)

if overrides:
    override_names = [FEATURE_LABELS.get(k, k) for k in overrides]
    st.caption(f"*Overrides: {', '.join(override_names)}*")

# --- Row 2: SHAP waterfalls side by side ---
col_shap1, col_shap2 = st.columns(2)

with col_shap1:
    fig_form = make_shap_waterfall(
        models["explainer_form"],
        features[MODEL_FEATURES["form"]],
        "Form (Building Energy)",
    )
    st.pyplot(fig_form, use_container_width=True)
    plt.close(fig_form)

with col_shap2:
    fig_mob = make_shap_waterfall(
        models["explainer_mobility"],
        features[MODEL_FEATURES["mobility"]],
        "Mobility (Transport Energy)",
    )
    st.pyplot(fig_mob, use_container_width=True)
    plt.close(fig_mob)

# --- Row 3: Archetype comparison ---
archetype_results: dict[str, dict] = {}
morph_positions = {"Detached": 0.0, "Semi": 0.33, "Terraced": 0.67, "Flat": 1.0}
for dtype, pos in morph_positions.items():
    if dtype not in archetypes:
        continue
    a = interpolate_archetype(pos, archetypes)
    feat = build_feature_vector(a)
    f_kwh = float(models["model_form"].predict(feat[MODEL_FEATURES["form"]])[0])
    m_kwh = float(models["model_mobility"].predict(feat[MODEL_FEATURES["mobility"]])[0])
    archetype_results[dtype] = {"form": f_kwh, "mobility": m_kwh, "total": f_kwh + m_kwh}

fig_comp, ax = plt.subplots(figsize=(10, 3.2))

categories = list(archetype_results.keys()) + ["Your scenario"]
form_vals = [archetype_results[d]["form"] for d in archetype_results] + [form_kwh]
mob_vals = [archetype_results[d]["mobility"] for d in archetype_results] + [mobility_kwh]
bands = [assign_band(f + m, models["band_thresholds"]) for f, m in zip(form_vals, mob_vals)]

x = np.arange(len(categories))

# Y-axis labels: "Category  [Band]" with band as coloured text
y_labels = []
for cat, b in zip(categories, bands):
    y_labels.append(f"{cat}  [{b}]")

bars_form = ax.barh(x, form_vals, 0.5, label="Form (building)", color="#3498db", alpha=0.9)
bars_mob = ax.barh(x, mob_vals, 0.5, left=form_vals, label="Mobility (transport)", color="#e67e22", alpha=0.9)

bars_form[-1].set_edgecolor("#aaaaaa")
bars_form[-1].set_linewidth(2)
bars_mob[-1].set_edgecolor("#aaaaaa")
bars_mob[-1].set_linewidth(2)

ax.set_yticks(x)
# Plain labels on y-axis
ax.set_yticklabels([""] * len(categories))

# Draw custom y-axis labels: name + band badge, right-aligned to axis
for i, (cat, b) in enumerate(zip(categories, bands)):
    bc = BAND_COLORS.get(b, "#888888")
    # Category name — far left
    ax.text(-1800, i, cat, va="center", ha="right", fontsize=10, color="#aaaaaa")
    # Band badge — between name and bar
    ax.text(-800, i, f" {b} ", va="center", ha="center", fontsize=10,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bc, edgecolor="none", alpha=0.9))

# kWh total to the right of each bar
for i, (f, m) in enumerate(zip(form_vals, mob_vals)):
    total_val = f + m
    ax.text(total_val + 150, i, f"{total_val:,.0f} kWh",
            va="center", fontsize=9, color="#aaaaaa")

max_total = max(f + m for f, m in zip(form_vals, mob_vals))
ax.set_xlim(-2500, max_total * 1.2)
ax.set_xlabel("")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=9,
          framealpha=0.3)

plt.subplots_adjust(left=0.02, bottom=0.25)
st.pyplot(fig_comp, use_container_width=True)
plt.close(fig_comp)

# --- Collapsible info ---
with st.expander("About this tool", expanded=True):
    st.markdown(
        """
**How this works.** The NEPI rates neighbourhoods on two energy costs — Form (building
energy) and Mobility (transport energy) — predicted from planner-controllable inputs:
neighbourhood morphology, walkable service coverage, public transport access, and building
era. The primary slider moves all inputs together along the observed national gradient from
sprawling detached suburbs (Band E, ~25,000 kWh/hh/yr, 1.6 cars/hh) to compact urban flats
(Band B, ~16,000 kWh/hh/yr, 0.7 cars/hh). Individual overrides let you test targeted
interventions while SHAP values show which inputs drive each prediction.

Four XGBoost models with monotonic constraints predict Form energy, Mobility energy,
car ownership, and commute distance from 198,779 English Output Areas. Each model sees only
its causally relevant features: building era affects Form but not Mobility; transit access
affects Mobility but not Form.
"""
    )
