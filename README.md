# Urban Energy

A national OA-level study of how neighbourhood form (morphology, density, walkable
access) shapes household energy consumption in England, packaged as the **Neighbourhood
Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC,
computed from open data.

**Live tool:** <https://UCL.github.io/urban-energy/> *(the old A–G Atlas; its source has been
removed pending a fresh two-axis rebuild — see below).*

> **⏸ Current focus.** The live work is the **[argument](paper/argument.md)** (the canonical
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
[paper/argument.md](paper/argument.md)):

- **⚡ Energy** (kWh/household/year) — what a household *spends*: metered **heat** (DESNZ
  gas + electricity) + **car travel** (anchored to measured NTS mileage by rural-urban class).
- **🌳 Access** — what the place *gives back*: the **everyday amenities reachable over the road
  network within each household's own travel catchment** (cityseer over OS Open Roads), plus the
  walkable richness within 1,600 m — and, unlike nearest distance, it can report **zero**.
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

**Energy** — a detached neighbourhood spends **1.74× a flat's household energy**:

| kWh/household/year | Flat | Detached | gap |
| --- | ---: | ---: | ---: |
| Heat (metered) | 10,194 | 15,020 | 1.5× |
| Car travel (NTS-anchored) | 3,240 | 9,272 | 2.9× |
| **Total energy** | **13,674** | **23,832** | **1.74×** |

**Access** — measured over the road network (cityseer), read three ways. **Like-for-like**, at any
matched distance a flat reaches **4.5–9.5× more** everyday amenities (density). A detached home
matches that count **only by driving ~2.4× as far** — for **~2.9× the energy**, so **~2.9× the
access per kWh**. And on foot the gap is starkest: a flat reaches **6.8 of 7** everyday
destinations within a walk, a detached home **4.6** — **60% of detached neighbourhoods can't reach
a single GP on foot.** *Same amenities, ~3× the energy — and a far poorer doorstep.*

**Lock-in** — perfect optimisation (best-practice insulation + full electrification) closes
only ~54% of the energy gap: a residual **1.47×** survives (bigger homes + longer trips), and
the access deficit is **100% tech-immune**. Built form fixes demand for generations.

(Numbers and confidence tags: [paper/argument.md](paper/argument.md); reproduce with
`stats/lock_in.py` + `stats/access_profile.py`.)

---

## Deliverables

### Current focus

1. **The argument** — the canonical two-axis statement in
   [paper/argument.md](paper/argument.md): hypothesis, claims ladder, and every headline
   number with its confidence tag. This is the single source of truth.
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
| [paper/argument.md](paper/argument.md) | **The argument** — canonical two-axis statement (single source of truth) |
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
uv run python stats/lock_in.py                  # energy 1.74× → optimised 1.47×
uv run python stats/access_profile.py           # network ~2.9×/kWh + walkable richness ~10×
uv run python stats/form_size_decomposition.py  # heat vs dwelling/household size
```

Full reproduction recipe (raw downloads → analysis) is in
[REPRODUCTION.md](REPRODUCTION.md), driven by the orchestrator
(`uv run python -m urban_energy.pipeline doctor`).

---

## Status

Full status, open work, and scope decisions (KEEP / DEFER / CUT) live in
**[ROADMAP.md](ROADMAP.md)**. Headline state:

**Done:** the national OA dataset (~178k OAs); the two-axis frame ([paper/argument.md](paper/argument.md));
NTS-anchored car-travel energy, the lock-in quantification, the **network access** measure (cityseer
over OS Open Roads, full per-OA curve; like-for-like 4.5–9.5×, drivable rate ~2.9×/kWh, 60% of
detached with no GP on foot), and the heat-vs-size decomposition (`stats/`); storage centralised behind
`URBAN_ENERGY_DATA_DIR`; and an executable rebuild
orchestrator (`urban_energy.pipeline`). The old three-surface code and A–G Atlas were removed
from the tree in the migration (in git history, pending reevaluation).

**Current focus:** keeping the argument + processing pipeline watertight.

**⏸ Pending (next phase):** the paper ([PAPER.md](PAPER.md)); reevaluating the Atlas scoring +
planning models for the two-axis frame.

---

## License

GPL-3.0-only. Author: Gareth Simons.
