# Figure story and design

The figures must tell the whole argument on their own: someone who reads only the
abstract, walks the figures in order, and reads the conclusion should leave with the
case. This document scaffolds that story first, then specifies each figure (what it
shows, the form, the palette). The palette is validated by script, not by eye
(results in section 3).

---

## 1. The story in one line

> A neighbourhood should be judged by the life its energy reaches, not the energy
> it burns. On that measure compact form wins, and clean technology does not change
> the result.

Told as five acts. Each act is one or two figures; read in order they are the paper.

### Act 1 — The wrong question, the right question

- **F1 · The inversion.** A detached neighbourhood spends about 2.1× a flat's energy
  per dwelling and reaches about 1/27 of the amenities on foot. Spends more, gets
  less. The hook. *(slopegraph, two axes crossing.)*
- **F2 · The whole country.** Every one of 178,353 Output Areas plotted as energy
  spent against access reached, coloured by dwelling mix. The cloud slopes down and
  flats sit top-left, detached bottom-right. F1 is the national pattern, not a
  cherry-pick. *(scatter / density.)*

### Act 2 — The energy axis: what a household spends

- **F3 · The energy gradient.** Total energy per dwelling by type, split into heat
  and car travel. Detached 2.1×, and travel (3.1×) is the steeper part, heat (1.6×)
  the rest. *(stacked bar.)*
- **F4 · Form or family?** The heat gap decomposed: 1.60× as-is, 1.27× at equal
  family size, 1.17× at equal floor area. About 17% is the shape of the building
  alone. Answers the first reviewer question. *(decomposition waterfall.)*

### Act 3 — The access axis: what a household reaches

- **F5 · On the doorstep.** Within a 1.6 km walk, a flat neighbourhood reaches about
  5 GPs, 14 schools, 93 food outlets; a detached reaches about 0, 2, 6. The 27×
  made concrete, service by service. *(paired dumbbell per service.)*
- **F6 · Reach against distance.** Amenities reachable from a short walk out to a
  25 km drive. The gap is widest on foot (27×), still 11× at 25 km, and never
  closes. *(multi-line, log y.)*

### Act 4 — The rate: the verdict

- **F7 · Same reach, a third of the fuel.** At its own car catchment a detached
  reaches a similar count, but drives about 2.4× further and burns about 3× the fuel
  to do it, so per kilowatt-hour a flat returns about 3.9× the access. *(rate bar,
  with the catchment distance shown.)*

### Act 5 — Locked in: so what

- **F8 · Technology cannot fix it.** The energy gap under insulation, heat pumps,
  EVs, the CCC 2040 pathway, and full deployment. No lever closes much of it, heat
  pumps even widen it, and access does not move at all. *(scenario ladder.)*
- **F9 · The map of England.** The rate (or the energy gap) for every Output Area,
  nationally. The pattern is structural and everywhere, not a local quirk.
  *(national choropleth.)*
- **F10 · Inside one city.** A single city, its compact core against its sprawling
  edge: access and energy flip across a few kilometres. Makes the national pattern
  tangible. *(city choropleth, two-panel access vs energy.)*

### Supplementary — Is it real

- **S1 · The unit trap.** The gap per household, per person, per m²; only the
  form-controlled estimate is honest. *(grouped bar.)*
- **S2 · Survives re-zoning.** The gap at OA, LSOA, MSOA. *(dumbbell by scale.)*
- **S3 · Survives selection.** The Oster bound for total energy and heat.

---

## 2. Per-figure specification

Every figure is a compositional (method-D) pure-type prediction unless noted, so
the plotted number matches summary.md and cannot drift. Titles state the finding,
not the variable. Sources are the stats scripts; figures write to `paper/figures/`
as PNG (300 dpi) + PDF.

| Fig | Form | Encoding | Colour | Source |
|---|---|---|---|---|
| F1 inversion | two-axis slopegraph | energy rail ↔ access rail, Flat/Detached lines cross | dwelling slate poles | `argument_figures.py` |
| F2 country | scatter of 178k OAs (+ 2-D density) | x energy/dwelling, y amenities on foot (log), hue = dominant type | dwelling ramp | new `fig_scatter` |
| F3 energy | stacked bar | x type, stack heat/travel | heat + travel hues | `argument_figures.py` |
| F4 form/family | horizontal waterfall | 1.60→1.27→1.17×, each control's bite | heat hue + neutral steps | new, from `form_size` |
| F5 doorstep | paired dumbbell per service | Flat vs Detached count within 1.6 km, six services | access green vs slate | new, from network cache |
| F6 reach | multi-line, log y | x distance, y amenities, one line per type | dwelling ramp | `argument_figures.py` |
| F7 rate | bar + catchment annotation | amenities per kWh by type, trip distance labelled | access green | `argument_figures.py` |
| F8 scenarios | horizontal ladder | Det:Flat gap per lever vs parity + status quo | status pair | `argument_figures.py` |
| F9 England | choropleth | OA polygons, fill = rate or energy gap | sequential / diverging | `map_figures.py` |
| F10 city | two-panel choropleth | one city, access panel and energy panel | sequential (two hues) | `map_figures.py` |

---

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
  over a lighter mechanism line.
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
