"""
Shared figure style for the NEPI paper figures.

One module holding the validated palette, the matplotlib rcParams and small
helpers, imported by every figure script so the visual system cannot drift figure
to figure. The full rationale is in ``paper/figure_design.md``; both palettes below
passed the data-visualisation validator (semantic trio: worst adjacent CVD ΔE 51;
dwelling ramp: monotone ordinal, light end 2.57:1 on the light surface).

Colour is split into two non-overlapping roles so a reader never has to ask whether
a hue means a quantity or a dwelling type:

* **Semantic hues** — what the quantity is: heat, car travel, access.
* **Dwelling ramp** — which type: a neutral slate ordinal ramp, compact → dispersed.
* **Status pair** — scenario direction (closes / widens); the sign is also in the
  direct label, so colour never carries it alone.

All figures are written to :data:`FIG_DIR` (``paper/figures``) as a 300-dpi PNG and
a vector PDF.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from urban_energy.paths import PROJECT_DIR

# Output folder: every publication figure lands here (PNG + PDF).
FIG_DIR = PROJECT_DIR / "paper" / "figures"

# --- Semantic hues: what the quantity is ---
HEAT = "#c1543b"  # metered gas + electricity (warm)
TRAVEL = "#5b7c9d"  # NTS-anchored car energy (cool, slate-leaning so the pair
#                     with HEAT does not read as the default matplotlib red/blue)
ACCESS = "#3d8a5f"  # amenities / jobs / people reached (green)

# --- Dwelling-type ordinal ramp (slate): compact → dispersed ---
DWELLING: dict[str, str] = {
    "Flat": "#7d92a8",
    "Terraced": "#5d7189",
    "Semi": "#3f4f61",
    "Detached": "#232f3d",
}
DWELLING_ORDER = ["Flat", "Terraced", "Semi", "Detached"]

#: Display names for figure labels/legends (keys stay the internal names).
DWELLING_LABEL: dict[str, str] = {
    "Flat": "Flats",
    "Terraced": "Terraced",
    "Semi": "Semi-detached",
    "Detached": "Detached",
}

# --- Status pair: scenario direction (sign also stated in the label) ---
CLOSES = "#3d8a5f"
WIDENS = "#c98a2e"
NEUTRAL = "#8a8a8a"

# --- Sequential ramps: one hue, light → dark. Energy maps use the warm ramp,
# access maps the green ramp, so the two never read as one flipped blue scale. ---
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
WARM_SEQ = ["#f7ddd0", "#e6a07a", "#cf6a43", "#a83c1f", "#6e2410"]  # energy
GREEN_SEQ = ["#e2f0e7", "#a7d0b5", "#5aa877", "#2f7d4f", "#12532c"]  # access

# --- Ink / chrome ---
INK = "#111111"
INK_SECONDARY = "#52514e"
MUTED = "#8a8a8a"
BASELINE = "#c3c2b7"
GRID = "#e1e0d9"

# Shared left edge (figure fraction) for the deck and the footer.
LEFT_X = 0.07

# Print widths (inches): single- and double-column journal sizes (~84 / 174 mm).
COL1 = 3.30
COL2 = 6.85


def apply_style() -> None:
    """Set the shared matplotlib rcParams. Call once before building figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: Figure, name: str, pdf: bool = True) -> Path:
    """Write ``fig`` to the output folder as a 300-dpi PNG (plus vector PDF).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (no extension); ``paper/figures/<name>.png``.
    pdf : bool, default False
        Also write a vector PDF. Off by default: a single PNG per figure keeps the
        output folder simple (and choropleths make a vector PDF enormous). Turn on
        for the camera-ready vector export.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=300)
    if pdf:
        fig.savefig(FIG_DIR / f"{name}.pdf")
    return png


# Editorial accent (figure kickers) and the standing source line for every footer.
ACCENT = "#2f6b49"  # deep green brand accent, distinct from the data hues
SOURCE = (
    "178,353 English Census 2021 Output Areas.  Metered DESNZ energy, "
    "NTS-anchored car travel, cityseer network access.\n"
    "Compositional pure-type estimates unless noted."
)


def deck(ax, kicker: str, title: str, subtitle: str | None = None) -> None:
    """Editorial title block above the axes: kicker, finding, mechanism line.

    Set the environment variable ``NEPI_PLAIN_FIGS=1`` to suppress the block
    (journal-submission variant: captions carry the words instead).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to title.
    kicker : str
        Short section label (rendered uppercase, accent colour), e.g. "The rate".
    title : str
        The finding, stated plainly (bold, primary ink).
    subtitle : str or None
        A lighter second line naming the mechanism or the caveat.
    """
    if os.environ.get("NEPI_PLAIN_FIGS") == "1":
        return
    y_kicker = 46 if subtitle else 32
    y_title = 24 if subtitle else 10
    # x anchors to the FIGURE margin (shared with the footer) so the deck sits
    # on one left edge across the set, however wide a plot's y-labels are;
    # y stays on the axes so the deck tracks the plot top.
    deck_coords = ("figure fraction", "axes fraction")
    ax.annotate(
        kicker.upper(),
        xy=(LEFT_X, 1),
        xytext=(0, y_kicker),
        xycoords=deck_coords,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=ACCENT,
    )
    ax.annotate(
        title,
        xy=(LEFT_X, 1),
        xytext=(0, y_title),
        xycoords=deck_coords,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(LEFT_X, 1),
            xytext=(0, 8),
            xycoords=deck_coords,
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=INK_SECONDARY,
        )


def footer(fig: Figure, text: str = SOURCE) -> None:
    """Add a small muted source line below the figure canvas.

    Anchored beneath y=0 so the tight save box grows to include it: the
    footer can never collide with an axis title, whatever the margins.
    """
    if os.environ.get("NEPI_PLAIN_FIGS") == "1":
        return
    fig.text(LEFT_X, 0.0, text, ha="left", va="top", fontsize=7, color=MUTED)


def comma(ax, which: str = "both") -> None:
    """Thousands-separate the numeric tick labels on the given axis/axes."""
    import matplotlib.ticker as mticker

    fmt = mticker.FuncFormatter(lambda v, _pos: f"{v:,.0f}")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
