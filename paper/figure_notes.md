# Figure notes — story, design system, review status

Merged from the former `figure_design.md` (story + specification) and
`figure_review_notes.md` (expert critique status). The figures must tell the whole
argument on their own: someone who reads only the abstract, walks the figures in
order, and reads the conclusion should leave with the case.

## 1. The story in one line

> A neighbourhood should be judged by the life its energy reaches, not the energy
> it burns. On that measure compact form wins, and clean technology does not change
> the result.

Told as five acts, plus two synthesis figures (F11, F12).

- **Act 1 — the wrong question, the right question.** F1 the inversion (spends 2.1×,
  reaches 1/27); F2 the whole country (178k OAs, the national cloud).
- **Act 2 — the energy axis.** F3 the energy gradient (heat + travel stacked);
  F4 form or family (1.60→1.27→1.17×, decomposition waterfall).
- **Act 3 — the access axis.** F5 on the doorstep (service-by-service dumbbells);
  F6 reach against distance (27× on foot, 11× at 25 km, never closes).
- **Act 4 — the rate.** F7 same reach, a third of the fuel (3.9× access per kWh).
- **Act 5 — locked in.** F8 technology cannot fix it (scenario ladder, both axes);
  F9 the map of England; F10 inside one city.
- **Synthesis.** F11 headline forest (all ratios, stem-and-dot, two panels on
  independent log rulers); F12 access and deprivation (decile bars over the city
  ladder of Spearman ρ).

## 2. Per-figure specification

Every figure is a compositional (method-D) pure-type prediction unless noted, so
the plotted number matches summary.md and cannot drift. Titles state the finding,
not the variable. Sources are the stats scripts; figures write to `paper/figures/`
as PNG (300 dpi) + PDF.

| Fig | Form | Encoding | Colour | Source |
|---|---|---|---|---|
| F1 inversion | two-axis slopegraph | energy rail ↔ access rail, Flat/Detached lines cross | dwelling slate poles | `argument_figures.py` |
| F2 country | hexbin density of 178k OAs | x energy/dwelling, y amenities on foot, hue = dominant type | dwelling ramp | `argument_figures.py` |
| F3 energy | stacked bar | x type, stack heat/travel | heat + travel hues | `argument_figures.py` |
| F4 form/family | horizontal waterfall | 1.60→1.27→1.17×, each control's bite | heat hue + neutral steps | `argument_figures.py` |
| F5 doorstep | paired dumbbell per service, symlog x | Flat vs Detached count within 1.6 km, six services | access green vs slate | `argument_figures.py` |
| F6 reach | multi-line, log y | x distance, y amenities, one line per type | dwelling ramp | `argument_figures.py` |
| F7 rate | bar + catchment annotation | amenities per kWh by type, trip distance labelled | access green | `argument_figures.py` |
| F8 scenarios | two-panel ladder | energy gap narrowing left, access frozen at 27× right | status pair | `argument_figures.py` |
| F9 England | choropleth | OA polygons, fill = rate or energy gap | sequential / diverging | `map_figures.py` |
| F10 city | two-panel choropleth | one city, access panel and energy panel | sequential (two hues) | `map_figures.py` |
| F11 forest | stem-and-dot, two panels | per estimate: stem from 1×, CI band, marker at tip | series hues | `argument_figures.py` |
| F12 equity | decile bars + city ladder | walkable access vs income deprivation; ρ per city, marker area ∝ OA count | access green / neutral | `argument_figures.py` |

## 3. The visual system

### 3.1 Two colour roles, kept separate (validated)

- **Semantic hues — what the quantity is.** Heat `#c1543b`, car travel `#3b6ea5`,
  access `#3d8a5f`. Validated categorical: worst adjacent CVD ΔE 51.1 (deutan), all
  in band, all ≥ 3:1.
- **Dwelling ordinal ramp — which type.** Slate, compact→dispersed: Flat `#93a1b0`,
  Terraced `#63758a`, Semi `#41505f`, Detached `#232f3d`. Validated ordinal:
  monotone, single hue, light end 2.57:1.
- **Status pair — scenario direction.** Closes `#3d8a5f`, widens `#c98a2e`,
  status-quo grey `#8a8a8a`; sign also in the label.
- **Sequential (maps).** Blue `#cde2fb → #0d366b`, or diverging blue ↔ heat brick
  with a light-grey midpoint for above/below median.

### 3.2 Craft

- **Type**: Helvetica Neue throughout, one face. Deck titles: a bold finding line
  over a lighter mechanism line (suppressible with `NEPI_PLAIN_FIGS=1` at submission).
- **Marks**: 2 px lines, ≥ 8 px markers, a 2 px surface gap between adjacent fills,
  recessive hairline grid on the value axis only, top/right spines off.
- **Labels over legends**: direct-label the two poles (Flat, Detached); a legend
  only when four series need naming; never a number on every mark.
- **Dimensions**: single-column 3.3 in, double-column 6.85 in, 300 dpi PNG + vector
  PDF.
- **Captions**: one plain sentence, the finding and the number, scanned against the
  banned-AI-speak list. No em-dashes.

### 3.3 Build

One style module, `stats/figstyle.py`, holds the palette, rcParams and helpers, so
the system cannot drift figure to figure. Each figure prints the number it encodes,
so the caption and the plot cannot diverge. On any hue change, re-check contrast
against the palette rules above (the one-off validator script was not kept).

## 4. Review status (three expert critiques, 2026-07-09)

Applied in the rebuild: F8 drawn on both axes; F10 two-hue maps; colour discipline +
shared editorial frame on every figure; F7 middle types greyed with the 3.9×
bracketed; F5 symlog; F2 hexbin; F6 both reference lines; F4 proper waterfall;
F9 light land + city labels; the named label collisions; one full-width PNG per
figure. Later (2026-07-30): F9/F10 panel titles renamed, F11 rebuilt as stem-and-dot,
F12 redesigned (decile bars + city ladder) and signed off.

Open, to pick up if desired:

- **Sequence**: the deck ends on a map; consider ending on F8 (the payoff) with the
  maps moved to the F2 "it's national" beat or supplementary.
- **F4**: consider demoting to supplementary (a rigour detour mid-arc), or retitle
  as a positive finding.
- **F1 title**: put the numbers in ("burns 2.1× the energy to reach 1/27 the
  amenities").
- **F3**: desaturate the travel blue toward slate; per-segment value labels.
- **Minor collisions**: F8 "today" label near the 2.2 tick; F2 leader label may clip;
  F10 source footer close to the colourbars.
- **Provenance**: the amenity source/filter table lives in `summary.md` (Access
  section); keep in sync if any layer filter changes.
