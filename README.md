# Urban Energy

A national OA-level study of how neighbourhood form (morphology, density, walkable
access) shapes household energy consumption in England, packaged as the **Neighbourhood
Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC,
computed from open data.

**Live tool:** <https://pub-e464ff17413e4256adbd9f89496bad9c.r2.dev/index.html>
*(experimental demo of the rebuilt two-axis Atlas — under development and pending review;
grades and numbers may change).*

> **⏸ Current focus.** The live work is the **manuscript ([paper/latex/main.tex](paper/latex/main.tex))**,
> written on the two-axis frame and prepared for journal submission, and the
> **data + analysis pipeline**. The **Atlas has been rebuilt** on the two-axis frame and
> soft-launched as the demo above; the XGBoost planning models are dropped from the plan
> (their code stays in git history). The theory + headline below are the current two-axis frame.

---

## The theory in 60 seconds

Cities are conduits that capture energy and recycle it through layers of human interaction
(Jacobs, 2000). The measure of urban energy efficiency is **not how much energy a
neighbourhood consumes, but how many transactions, connections, and functions that energy
enables before it dissipates.** A dense neighbourhood, like a rainforest, passes energy
through multiple trophic layers — street network, commercial exchange, public transport,
green space — each capturing value from the layer below. A sprawling suburb, like a
desert, dissipates the same energy in a single pass.

This connects to Bettencourt et al. (2007): cities scale superlinearly in socioeconomic
output (~N^1.15) and sublinearly in infrastructure (~N^0.85). The mechanism is **proximity**.

Three established empirical regularities converge:

1. **Building physics** — compact dwelling types have lower surface-to-volume ratios and
   share party walls, reducing heat loss per unit floor area (Rode et al., 2014).
2. **Transport geography** — Newman & Kenworthy (1989) showed the inverse density–fuel
   relationship; Ewing & Cervero (2010) and Stevens (2017) refined it: **destination
   accessibility** matters more than density alone.
3. **Metered vs modelled energy** — Few et al. (2023) showed EPC SAP estimates
   systematically over-predict consumption, so we use DESNZ postcode-level metered data
   to sidestep the performance gap.

NEPI puts this on **two measured axes** and a **rate** (the canonical statement is
[paper/summary.md](paper/summary.md)):

- **⚡ Energy** (kWh/household/year) — what a household *spends*: metered **heat** (DESNZ
  gas + electricity) + **car travel** (anchored to measured NTS mileage by rural-urban class).
- **🌳 Access** — what the place *gives back*: the **everyday amenities reachable over the road
  network within each household's own travel catchment** (cityseer over OS Open Roads), plus what
  is reachable on foot within 1,600 m — and, unlike nearest distance, it can report **zero**.
- **📐 The rate** = access ÷ energy. *The measure of a place is not how much energy it
  consumes, but how much access that energy buys.*

The analysis is descriptive and ecological (Robinson, 1950; Greenland, 2001): morphology is
genuinely an area-level property, so the ecological design is the correct level of analysis,
not a limitation. **The empirical result: insulation and fleet electrification can compress
the energy gap on technology-replacement timescales, but the access deficit is set by street
layout and turns over on generational timescales — even fully decarbonised, sprawl delivers
less access per Joule.** This is the carbon/infrastructure lock-in (Seto et al. 2016; Unruh 2000).

---

## Headline result (~178k OAs, England)

**Energy** — a detached neighbourhood spends about **2.1× a flat's energy per dwelling**:

| kWh/dwelling/year | Flat | Detached | gap (flat→detached) |
| --- | ---: | ---: | ---: |
| Heat (metered) | 10,194 | 15,020 | 1.6× |
| Car travel (NTS-anchored) | 3,240 | 9,272 | 3.1× |
| **Total energy** (per-OA median) | **13,674** | **23,832** | **2.1×** |

The Flat/Detached columns are observed medians; the gap is the compositional flat-to-detached
estimate per dwelling, so it is not the column quotient. Energy is modelled per dwelling with
family size and floor area held as free controls — not divided per person, which would impose a
household-size elasticity of 1 when heat's is about 0.5 ([paper/summary.md](paper/summary.md)).

**Access** — measured over the road network (cityseer). On foot a flat reaches about **27× the
amenities, 52× the jobs and 12× the people** of a detached neighbourhood; even at a 25 km drive the
flat is still **11–14× ahead**. At each area's own car catchment the raw counts nearly converge: a
detached area gets there only by driving much further, so per kilowatt-hour a flat returns about
**3.9× the access** a detached home does.

**Lock-in** — no decarbonisation lever closes much of the energy gap, and none moves access at all.
Taken separately, insulation closes about a fifth of the gap, heat pumps leave it marginally wider (a
delivered-energy fuel switch that unmasks car travel), electric vehicles close about a fifth. The
CCC's 2040 Balanced Pathway leaves **1.89×** and a full rollout of all three leaves **1.68×**, about
two-thirds surviving, while the access deficit is **100% unchanged**. Fabric plus full electrification
without heat pumps is the conventional bound, **2.12× → 1.51×**. Built form fixes demand for generations.

(Full numbers and method: [paper/summary.md](paper/summary.md); reproduce with
`stats/scenarios.py` + `stats/lock_in.py` + `stats/access_profile.py`.)

---

## Deliverables

### Current focus

1. **The manuscript** — [paper/latex/main.tex](paper/latex/main.tex) (prepared for submission,
   with [paper/latex/extended_data.tex](paper/latex/extended_data.tex)). Every result number in
   it is a `\nepi` macro written by the stats scripts through `stats/ledger.py`, so the
   manuscript regenerates with the analysis; see
   [paper/submission_checklist.md](paper/submission_checklist.md) for the recipe and state.
2. **The data + analysis pipeline** — acquisition orchestrator + the two-axis analysis layer
   (`oa_data` + `oa_access` → `travel_energy`, `access_profile`, `lock_in`, `form_size`),
   reproducible from open data with no heavy processing step.
3. **The NEPI Atlas** — built on the two-axis frame: `stats/nepi_score.py` (A–G score,
   bands frozen at 2021) + `stats/atlas_export.py` → the static site in `site/`,
   soft-launched at the live-tool link above; full launch on acceptance
   ([dissemination/launch_checklist.md](dissemination/launch_checklist.md)).

---

## Project structure

| Path | Purpose |
| ---- | ------- |
| [paper/latex/main.tex](paper/latex/main.tex) | **The manuscript** — prepared for submission; result numbers ledger-wired via `stats/ledger.py` |
| [paper/summary.md](paper/summary.md) | The argument — narrative two-axis statement (companion to the manuscript) |
| [CLAUDE.md](CLAUDE.md) | **Technical brief** — codebase layout, data, architecture, conventions |
| [REPRODUCTION.md](REPRODUCTION.md) | **How to rebuild** — orchestrator-driven recipe, manual downloads |
| [ROADMAP.md](ROADMAP.md) | **Status, scope & open work** — incl. the methodology decisions |
| [paper/literature_review.md](paper/literature_review.md) | Thematic literature review |
| [paper/references.bib](paper/references.bib) | BibTeX bibliography (partial) |
| [data/](data/) | Raw-data acquisition and preprocessing scripts |
| [stats/](stats/) | Two-axis analysis: `oa_data` core + travel energy, access profile, lock-in, form/size |

The `data/` and `stats/` directories contain code only — see
[CLAUDE.md](CLAUDE.md) for the full inventory of scripts and outputs.

---

## Quick start

```bash
# Install + configure
uv sync
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# Two-axis analysis — energy gradient, scenarios, access profile, form/size
uv run python stats/oa_network_access.py        # build network-access cache (cityseer, ~12 min)
uv run python stats/lock_in.py                  # fabric+EV bound 2.12× → 1.51× (per dwelling)
uv run python stats/scenarios.py                # scenario ladder: fabric/heat-pump/EV separate levers, CCC pathway
uv run python stats/maup_scale.py               # MAUP: gap re-fit at OA/LSOA/MSOA (2.12/1.88/1.72×)
uv run python stats/access_profile.py           # access per kWh 3.9×, on-foot gap ~27×
uv run python stats/form_size_decomposition.py  # heat 1.60× → 1.17× size-held (family size a free control, γ≈0.5)
```

Full reproduction recipe (raw downloads → analysis) is in
[REPRODUCTION.md](REPRODUCTION.md), driven by the orchestrator
(`uv run python -m urban_energy.pipeline doctor`).

---

## Status

Full status, open work, and scope decisions (KEEP / DEFER / CUT) live in
**[ROADMAP.md](ROADMAP.md)**. Headline state:

**Done:** the national OA dataset (~178k OAs); the two-axis frame ([paper/summary.md](paper/summary.md));
NTS-anchored car-travel energy, the lock-in quantification, the **network access** measure (cityseer
over OS Open Roads, full per-OA curve; on-foot gap ~27×, drivable rate 3.9× access per kWh), and the
heat-vs-size decomposition (`stats/`), all on a compositional flat-vs-detached estimator; storage centralised behind
`URBAN_ENERGY_DATA_DIR`; and an executable rebuild
orchestrator (`urban_energy.pipeline`); the NEPI score + Atlas rebuilt on the two-axis frame
and soft-launched (`stats/nepi_score.py`, `stats/atlas_export.py`, `site/`). The old
three-surface code and the XGBoost planning models stay in git history; XGBoost is dropped
from the plan.

**Current focus:** the editorial revision of the manuscript, then submission
([paper/submission_checklist.md](paper/submission_checklist.md)).

---

## License

GPL-3.0-only. Author: Gareth Simons.
