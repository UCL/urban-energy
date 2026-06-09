# Urban Energy

A national OA-level study of how neighbourhood form (morphology, density, walkable
access) shapes household energy consumption in England, packaged as the **Neighbourhood
Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC,
computed from open data.

**Live tool:** <https://UCL.github.io/urban-energy/>

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

Three dimensions, three policy silos, no integrated metric — until the **NEPI**. We rate
each Output Area on three surfaces, all in **kWh/household/year** so the composite needs
no arbitrary weighting:

- **Form** — DESNZ metered building energy
- **Mobility** — Census commute × ECUK energy intensities
- **Access** — empirical OLS estimate of the *additional* transport energy attributable
  to poor walkable service coverage, relative to a compact reference (85% coverage)

The composite is banded **A–G** by national percentile, directly analogous to a building
EPC. Where an EPC rates the dwelling envelope, the NEPI rates the place.

The analysis is descriptive and ecological (Robinson, 1950; Greenland, 2001): morphology
is genuinely an area-level property, so the ecological design is the correct level of
analysis, not a limitation. **The empirical result: building retrofit and fleet
electrification can compress the Form and Mobility gaps on technology-replacement
timescales, but the Access gap is set by street layout and turns over on generational
timescales — and that is the surface no current policy addresses.**

---

## Headline result (6,687 BUAs / 198,779 OAs)

| | Flat-dominant OA | Detached-dominant OA | Gap |
|---|---:|---:|---:|
| **Form** (building energy) | 10,755 kWh/hh | 15,713 kWh/hh | 1.46× |
| **Mobility** (transport, overall) | 4,150 kWh/hh | 9,185 kWh/hh | 2.21× |
| **Access penalty** (empirical OLS) | 0 | 1,519 kWh/hh | — |
| **Total NEPI** | **15,982 (Band A)** | **26,897 (Band F)** | **+10,915** |
| kWh per unit access | 3,292 | 8,820 | 2.68× |

Decomposition of the 10,915 kWh/hh/yr gap: **Form 45% / Mobility 43% / Access 14%**.

**Robustness:** the gradient *steepens* at stricter plurality thresholds (1.84× at 60%
purity), and the pre-pandemic Census 2011 transport gradient is **steeper** than the
COVID-affected 2021 figure (2.00× vs 1.70×). Full robustness section in
[PAPER.md §5](PAPER.md).

---

## Three deliverables

1. **The paper** — full IMRaD case in [PAPER.md](PAPER.md), with §5 robustness section
   already drafted. Targets a peer-reviewed journal.
2. **The NEPI Atlas** — a public, place-level A–G energy-rating dashboard, live on
   GitHub Pages: <https://UCL.github.io/urban-energy/> (source in [stats/atlas/](stats/atlas/),
   exported to [stats/nepi_static/](stats/nepi_static/) and mirrored to [docs/](docs/)).
3. **The NEPI planning tool** — four monotonically-constrained XGBoost models (form /
   mobility / cars / commute), embedded in the Atlas as in-browser HTML/JS and also
   available as a Streamlit app ([stats/nepi_app.py](stats/nepi_app.py)) with SHAP waterfalls.

---

## Project structure

| Path | Purpose |
| ---- | ------- |
| [PAPER.md](PAPER.md) | **The paper** — full IMRaD case (canonical) |
| [CLAUDE.md](CLAUDE.md) | **Technical brief** — codebase, data, scripts, conventions, repro |
| [paper/literature_review.md](paper/literature_review.md) | Thematic literature review |
| [paper/references.bib](paper/references.bib) | BibTeX bibliography (partial) |
| [data/](data/) | Raw-data acquisition and preprocessing scripts |
| [processing/](processing/) | National OA pipeline (`pipeline_oa.py` — CityNetwork API, all 7,147 BUAs) |
| [stats/](stats/) | Case figures, NEPI scorecard, access penalty model, planning tool |
| [docs/](docs/) | GitHub Pages mirror of [stats/nepi_static/](stats/nepi_static/) |
| [notes/](notes/) | Archived v0 working notes (LSOA-era snapshots) |
| [paper/archive/](paper/archive/) | LSOA case_v1 + stale LaTeX |

The `data/`, `processing/`, and `stats/` directories contain code only — see
[CLAUDE.md](CLAUDE.md) for the full inventory of scripts and outputs.

---

## Quick start

```bash
# Install + configure
uv sync
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# Regenerate all OA case figures + tables
uv run python stats/build_case_oa.py

# NEPI scorecard, bands, surface decomposition
uv run python stats/nepi.py
uv run python stats/access_penalty_model.py

# Interactive planning tool
uv run streamlit run stats/nepi_app.py
```

Full reproduction recipe (raw downloads → national pipeline → trained models → Atlas)
is in [REPRODUCTION.md](REPRODUCTION.md), driven by the orchestrator
(`uv run python -m urban_energy.pipeline doctor`).

---

## Status

Full status, open work, and the 2026-06-09 lean-pipeline scope decisions
(KEEP / DEFER / CUT) live in **[ROADMAP.md](ROADMAP.md)**. Headline state:

**Done:** national OA pipeline (198,779 OAs), NEPI scorecard + A–G bands + surface
decomposition, empirical access-penalty OLS, four monotonic XGBoost models + SHAP,
Streamlit + static HTML/JS tool live on GitHub Pages, the IMRaD case ([PAPER.md](PAPER.md)),
storage centralised behind `URBAN_ENERGY_DATA_DIR`, methodology #6 Form under-recording
flags, and an executable rebuild orchestrator (`urban_energy.pipeline`).

**Open:** see [ROADMAP.md](ROADMAP.md) — analysis decisions (per-capita unit, SHAP
interpretation, lock-in framing), paper finalisation, and deferred LiDAR/morphology.

---

## License

GPL-3.0-only. Author: Gareth Simons.
