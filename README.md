# Urban Energy

A national OA-level study of how neighbourhood form (morphology, density, walkable
access) shapes household energy consumption in England, packaged as the **Neighbourhood
Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC,
computed from open data.

**Live tool:** <https://UCL.github.io/urban-energy/> *(the old A–G Atlas; its source has been
removed pending a fresh two-axis rebuild — see below).*

> **⏸ Current focus.** The live work is the **[argument](paper/summary.md)** (the canonical
> two-axis statement) and the **data + analysis pipeline**. The **paper ([PAPER.md](PAPER.md)) is
> deferred**, and the **Atlas is pending** — its scoring and the XGBoost planning models are to
> be reevaluated for the two-axis frame (that code lives in git history). The theory + headline
> below are the current two-axis frame.

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
| **Total energy** | **13,674** | **23,832** | **2.1×** |

The Flat/Detached columns are observed medians; the gap is the compositional flat-to-detached
estimate per dwelling, so it is not the column quotient. Energy is modelled per dwelling with
family size and floor area held as free controls — not divided per person, which would impose a
household-size elasticity of 1 when heat's is about 0.5 ([paper/summary.md](paper/summary.md)).

**Access** — measured over the road network (cityseer). On foot a flat reaches about **24× the
amenities, 52× the jobs and 12× the people** of a detached neighbourhood; even at a 25 km drive the
flat is still **10–14× ahead**. At each area's own car catchment the raw counts nearly converge: a
detached area gets there only by driving much further, so per kilowatt-hour a flat returns about
**3.6× the access** a detached home does.

**Lock-in** — best-practice insulation + full electrification closes only about half the energy
gap: per dwelling **2.12× → 1.51×** (at equal family size 1.71× → 1.18×), the residual being bigger
homes and longer trips, while the access deficit is **100% unchanged**. Built form fixes demand for generations.

(Full numbers and method: [paper/summary.md](paper/summary.md); reproduce with
`stats/lock_in.py` + `stats/access_profile.py`.)

---

## Deliverables

### Current focus

1. **The argument** — the canonical two-axis statement in
   [paper/summary.md](paper/summary.md): the hypothesis, the method, and every headline
   number. This is the single source of truth.
2. **The data + analysis pipeline** — acquisition orchestrator + the two-axis analysis layer
   (`oa_data` + `oa_access` → `travel_energy`, `access_profile`, `lock_in`, `form_size`),
   reproducible from open data with no heavy processing step.

### ⏸ Pending (next phase)

1. **The paper** — deferred ([PAPER.md](PAPER.md)).
2. **The NEPI Atlas + planning tool** — pending: reevaluate the place-scoring and the XGBoost
   planning models for the two-axis frame (their code lives in git history).

---

## Project structure

| Path | Purpose |
| ---- | ------- |
| [paper/summary.md](paper/summary.md) | **The argument** — canonical two-axis statement (single source of truth) |
| [PAPER.md](PAPER.md) | The formal IMRaD paper — **⏸ deferred** (old three-surface framing) |
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

# Two-axis analysis — energy gradient, lock-in, access profile, form/size
uv run python stats/oa_network_access.py        # build network-access cache (cityseer, ~12 min)
uv run python stats/lock_in.py                  # energy gap 2.12× → optimised 1.51× (per dwelling)
uv run python stats/access_profile.py           # access per kWh 3.6×, on-foot gap ~24×
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
over OS Open Roads, full per-OA curve; on-foot gap ~24×, drivable rate 3.6× access per kWh), and the
heat-vs-size decomposition (`stats/`), all on a compositional flat-vs-detached estimator; storage centralised behind
`URBAN_ENERGY_DATA_DIR`; and an executable rebuild
orchestrator (`urban_energy.pipeline`). The old three-surface code and A–G Atlas were removed
from the tree in the migration (in git history, pending reevaluation).

**Current focus:** keeping the argument + processing pipeline watertight.

**⏸ Pending (next phase):** the paper ([PAPER.md](PAPER.md)); reevaluating the Atlas scoring +
planning models for the two-axis frame.

---

## License

GPL-3.0-only. Author: Gareth Simons.
