# Urban Energy

A national OA-level study of how neighbourhood form (morphology, density, walkable
access) shapes household energy consumption in England, packaged as the **Neighbourhood
Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC,
computed from open data.

**Live tool:** <https://UCL.github.io/urban-energy/> *(the old A–G Atlas; its source has been
removed pending a fresh two-axis rebuild — see below).*

> **⏸ Current focus.** The live work is the **[argument](paper/argument.md)** (the canonical
> two-axis statement) and the **processing pipeline**. The **paper ([PAPER.md](PAPER.md)) is
> deferred** (still the old framing), and the **Atlas has been removed** — it implemented the
> retired *three-surface / A–G* model and will be rebuilt fresh for the two-axis frame (git
> history holds the old code). The theory + headline below are the current two-axis frame.

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
- **🌳 Access** — what the place *gives back*: the **count of everyday services within a
  walkable (1,600 m) catchment** — and, unlike nearest distance, it can report **zero**.
- **📐 The rate** = access ÷ energy. *The measure of a place is not how much energy it
  consumes, but how much access that energy buys.*

The analysis is descriptive and ecological (Robinson, 1950; Greenland, 2001): morphology is
genuinely an area-level property, so the ecological design is the correct level of analysis,
not a limitation. **The empirical result: insulation and fleet electrification can compress
the energy gap on technology-replacement timescales, but the access deficit is set by street
layout and turns over on generational timescales — even fully decarbonised, sprawl delivers
less access per Joule.** This is the carbon/infrastructure lock-in (Seto et al. 2016; Unruh 2000).

---

## Headline result (~174k OAs with complete energy + form data)

**Energy** — a detached neighbourhood spends **1.78× a flat's household energy**:

| kWh/household/year | Flat | Detached | gap |
| --- | ---: | ---: | ---: |
| Heat (metered) | 10,196 | 15,462 | 1.5× |
| Car travel (NTS-anchored) | 3,239 | 9,073 | 2.8× |
| **Total energy** | **13,675** | **24,363** | **1.78×** |

**Access** — for the *same* energy, compact form buys **~10× more everyday access**
(geometric mean over eight services): 11× the GPs, 24× the shops, 20× the rail. **39% of
detached neighbourhoods have no GP within 1,600 m; 73% have no railway station.** *Pay more,
get less.*

**Lock-in** — perfect optimisation (best-practice insulation + full electrification) closes
only ~60% of the energy gap: a residual **1.44×** survives (bigger homes + longer trips), and
the access deficit is **100% tech-immune**. Built form fixes demand for generations.

(Numbers and confidence tags: [paper/argument.md](paper/argument.md); reproduce with
`stats/lock_in.py` + `stats/access_profile.py`.)

---

## Deliverables

### Current focus

1. **The argument** — the canonical two-axis statement in
   [paper/argument.md](paper/argument.md): hypothesis, claims ladder, and every headline
   number with its confidence tag. This is the single source of truth.
2. **The processing pipeline** — the national OA pipeline + the two-axis analysis layer
   (`stats/travel_energy.py`, `stats/access_profile.py`, `stats/lock_in.py`), reproducible
   end-to-end via the orchestrator.

### ⏸ Deferred / removed (next phase)

1. **The paper** — full IMRaD case in [PAPER.md](PAPER.md); deferred, to be rewritten to the
   two-axis frame once the argument + pipeline are locked.
2. **The NEPI Atlas + planning tool** — the old public A–G dashboard and the four XGBoost
   planning models implemented the retired three-surface framing and have been **removed**; a
   two-axis Atlas will be **rebuilt fresh** later (git history holds the old code).

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
| [processing/](processing/) | National OA pipeline (`pipeline_oa.py` — CityNetwork API) |
| [stats/](stats/) | Two-axis analysis: `oa_data` core + travel energy, access profile, lock-in, form/size |

The `data/`, `processing/`, and `stats/` directories contain code only — see
[CLAUDE.md](CLAUDE.md) for the full inventory of scripts and outputs.

---

## Quick start

```bash
# Install + configure
uv sync
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# Two-axis analysis — energy gradient, lock-in, access profile, form/size
uv run python stats/lock_in.py                  # energy now 1.78× → optimised 1.44×
uv run python stats/access_profile.py           # ~10× access per kWh, counts within 1,600 m
uv run python stats/form_size_decomposition.py  # heat vs dwelling/household size
```

Full reproduction recipe (raw downloads → national pipeline → analysis) is in
[REPRODUCTION.md](REPRODUCTION.md), driven by the orchestrator
(`uv run python -m urban_energy.pipeline doctor`).

---

## Status

Full status, open work, and scope decisions (KEEP / DEFER / CUT) live in
**[ROADMAP.md](ROADMAP.md)**. Headline state:

**Done:** national OA pipeline (~199k OAs); the two-axis frame ([paper/argument.md](paper/argument.md));
NTS-anchored car-travel energy, the lock-in quantification, the per-service access profile,
and the heat-vs-size decomposition (`stats/`); storage centralised behind
`URBAN_ENERGY_DATA_DIR`; methodology #6 Form under-recording flags; and an executable rebuild
orchestrator (`urban_energy.pipeline`). The retired three-surface code and the old A–G Atlas
were stripped in the two-axis migration.

**Current focus:** keeping the argument + processing pipeline watertight.

**⏸ Deferred (next phase):** rewrite the paper ([PAPER.md](PAPER.md)) to the two-axis frame;
rebuild the Atlas fresh; deferred LiDAR/morphology.

---

## License

GPL-3.0-only. Author: Gareth Simons.
