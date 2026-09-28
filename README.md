# Urban Energy

## ▶ NEPI Atlas demo: <https://pub-e464ff17413e4256adbd9f89496bad9c.r2.dev/index.html>

The Atlas is the Neighbourhood Energy Performance Index (NEPI): an A–G rating of every English neighbourhood on the energy its homes and cars use and the access that energy buys, searchable by postcode. The link above is an experimental demo. It is under development, not yet rebuilt on the audited data described below, and its grades may change.

---

The repository holds the data pipeline, analysis, and manuscript for a national study of England's neighbourhoods: how much energy their homes and cars use, how much access to everyday destinations their residents obtain, and how much of the energy premium of dispersed urban form survives decarbonisation. The manuscript is [paper/latex/main.tex](paper/latex/main.tex), *The energy premium of dispersed urban form and its persistence under decarbonisation*, with [paper/latex/extended_data.tex](paper/latex/extended_data.tex).

> **Current state (2026-09-28).** The manuscript is complete and in co-author review: 23 pages, 6 figures and 1 table in the main text, 5 figures and 6 tables in Extended Data. Every result number is a `\nepi` macro written by the stats scripts through `stats/ledger.py`, so the manuscript regenerates with the analysis. A full audit on 2026-09-11 reproduced the ledger from the code and data, corrected one aggregation step, and put every model on one analysis sample. Before submission: author list and affiliations, acknowledgements, and code and data DOIs ([paper/submission_checklist.md](paper/submission_checklist.md)).

---

## The question

Decarbonisation policy reduces the energy that homes and vehicles use through insulation, heat pumps, and electric vehicles. The energy a neighbourhood needs also depends on its form: compact neighbourhoods lose less heat through shared walls, and their residents drive less because destinations lie nearer. The paper asks whether the three measures, applied in full, can negate the energy premium of dispersed form, and answers it for the whole of England at neighbourhood resolution.

The premise is Jane Jacobs's account of ecosystems and economies as conduits for energy: the more compact, complex, and diverse a system, the more use it extracts from a unit of energy before that unit is lost. The paper names this the energy productivity principle and measures it at the neighbourhood, with energy as the unit. A compact neighbourhood should house more households for a given amount of energy and give them more access to amenities and services; the dispersed neighbourhood is Jacobs's desert, where the same energy yields a single use.

## The design

- **Unit.** The 2021 Census Output Area, about 125 households. The analysis covers the 178,322 English Output Areas with more than ten residents, positive metered consumption, a certificate-derived floor area and fabric ratio, and an allocated travel figure. Every model is fitted on this one sample.
- **Energy** (kWh per dwelling per year). Home energy is metered gas and electricity (DESNZ postcode statistics for 2024, aggregated to Output Areas). Car travel is National Travel Survey mileage by rural-urban class (NTS9904, 2024), disaggregated to Output Areas by car ownership and commute distance with the class totals preserved, and priced at the local fleet intensity.
- **Access.** The count of everyday destinations reachable over the road network (cityseer over OS Open Roads): a benchmark basket of six amenity types (GPs, pharmacies, schools, places to eat and drink, grocery shops, parks), plus jobs and people, read at 1,600 m, at each area's own car catchment, and at 25.6 km.
- **The rate.** Amenities within the car catchment divided by car-travel energy: the access a neighbourhood obtains per kilowatt-hour.
- **The comparison.** A compositional regression on each area's dwelling-type shares predicts what a neighbourhood built wholly of flats, terraced, semi-detached, or detached houses would consume and reach, with building age, deprivation, tenure, and climate held constant. Confidence intervals are clustered on the 309 local-authority districts.
- **Decarbonisation.** Energy is recomputed under insulation, heat pumps, and electric vehicles, alone and together, at the Climate Change Committee's 2040 Balanced Pathway uptakes and at full deployment, and the gap is re-estimated. Access is unchanged by construction.

## Headline result

**Energy.** A detached-type neighbourhood uses 2.11× (95% CI 2.00–2.23) the energy of a flat-type one per dwelling, or 1.70× at equal household size. Home energy accounts for 1.59× of that and car travel for 3.07×.

| kWh per dwelling per year | Flats | Detached | pure-type gap |
| --- | ---: | ---: | ---: |
| Home energy (metered) | 10,253 | 15,070 | 1.59× |
| Car travel (NTS-anchored) | 3,240 | 9,272 | 3.07× |
| **Total** (per-area median) | **13,735** | **23,903** | **2.11×** |

The columns are medians of the areas whose dominant dwelling type is flats or detached houses; the gap is the compositional estimate for the pure types with the confounds held, so it is not the column quotient. Energy is modelled per dwelling, with household size and floor area entered as controls whose strength is estimated from the data rather than fixed by a per-person or per-square-metre denominator.

**Access.** On foot, a flat-type neighbourhood reaches 27× the amenities, 52× the jobs, and 12× the people of a detached-type one, and 11× the amenities at a 25.6 km drive. At each type's own car catchment the counts nearly converge (1.26×), because residents of dispersed areas drive further to reach them, at 3.07× the car energy. Per kilowatt-hour a flat-type neighbourhood therefore obtains 3.9× (3.1–4.8) the access.

**Lock-in.** Insulation alone brings the total gap from 2.11× to 1.82×, electric vehicles alone to 1.84×, and heat pumps alone widen it to 2.16×, because they cut home energy near-uniformly and leave the larger travel gap showing. The CCC's 2040 pathway leaves 1.88×. Full deployment of all three measures leaves 1.67× (1.61–1.73), or 1.35× at equal household size, so 69% of the gap on the logarithmic scale survives, and the access gap is unchanged in every scenario. The surviving gap is wider than the variation among like neighbourhoods (an interquartile factor of 1.25), so the two types remain separate bands after full treatment.

**The premium in plain units.** After full deployment a flat-type neighbourhood runs on about 5,800 kWh per dwelling a year and a detached-type one on 9,600, a difference of 3,800 kWh. Across England's stock that is 36 TWh a year, a fifth of what the treated stock would use (13 TWh net of household size). Every 100,000 dwellings built as detached-type rather than flat-type neighbourhoods add 0.4 TWh a year for their lifetime.

**Equity.** The most income-deprived decile of areas holds 3.9× the walkable access of the least deprived, because deprivation is concentrated in compact form; the gradient inverts in the strongest housing markets (inner London, Manchester, Bristol, Cambridge, Oxford).

**Robustness.** The total gap withstands unrecorded sorting as strong as all the measured controls together (Oster δ* ≈ 1.2); it stays at 2.36× when car mileage is allocated at the class average; charging public-transport energy to residents leaves it at 1.99×; re-fitting at LSOA and MSOA scale gives 1.87× and 1.71×; clustering on fifty spatial blocks widens the interval to 1.89–2.37.

Full numbers with intervals are in the manuscript tables and [paper/results_snapshot.txt](paper/results_snapshot.txt); the narrative companion is [paper/summary.md](paper/summary.md).

---

## Deliverables

1. **The manuscript**, [paper/latex/main.tex](paper/latex/main.tex) and [paper/latex/extended_data.tex](paper/latex/extended_data.tex), ledger-wired through `stats/ledger.py`; state and remaining steps in [paper/submission_checklist.md](paper/submission_checklist.md).
2. **The data and analysis pipeline**: an acquisition orchestrator over open data (`urban_energy.pipeline`) and the analysis layer in `stats/`, reproducible end to end ([REPRODUCTION.md](REPRODUCTION.md)).
3. **The NEPI Atlas**: `stats/nepi_score.py` (A–G score on the rate, bands frozen at 2021) and `stats/atlas_export.py` → `site/`, soft-launched at the link above; rebuild on the audited data and full launch follow acceptance ([dissemination/launch_checklist.md](dissemination/launch_checklist.md)).
4. **Further papers** in [papers/](papers/): an energy-allometry prospectus with pilot scripts, and the outline of the NEPI score paper.

---

## Project structure

| Path | Purpose |
| ---- | ------- |
| [paper/latex/main.tex](paper/latex/main.tex) | **The manuscript**; result numbers ledger-wired via `stats/ledger.py` |
| [paper/latex/extended_data.tex](paper/latex/extended_data.tex) | Extended Data (5 figures, 6 tables) |
| [paper/summary.md](paper/summary.md) | Narrative companion to the manuscript |
| [paper/prose_guide.md](paper/prose_guide.md) | House prose style for every reader-facing text in the repository |
| [paper/results_snapshot.txt](paper/results_snapshot.txt) | Verbatim output of every ledger-writing script, last regenerated 2026-09-11 |
| [CLAUDE.md](CLAUDE.md) | **Technical brief**: codebase layout, data, architecture, conventions |
| [REPRODUCTION.md](REPRODUCTION.md) | **How to rebuild**: orchestrator-driven recipe, manual downloads |
| [ROADMAP.md](ROADMAP.md) | Status, scope, and open work, including the methodology decisions |
| [paper/literature_review.md](paper/literature_review.md) | Thematic literature review |
| [paper/references.bib](paper/references.bib) | BibTeX bibliography |
| [data/](data/) | Raw-data acquisition and Output Area aggregation scripts |
| [stats/](stats/) | Analysis: `oa_data` core, travel energy, access, scenarios, robustness, figures, ledger |
| [dissemination/](dissemination/) | NEPI score specification, Atlas architecture, launch checklist, frozen bands |

The `data/` and `stats/` directories contain code only; the built artefacts live under `$URBAN_ENERGY_DATA_DIR` (see [CLAUDE.md](CLAUDE.md)).

---

## Quick start

```bash
# Install and configure
uv sync
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# Acquire the open data and aggregate to Output Areas
uv run python -m urban_energy.pipeline doctor
uv run python -m urban_energy.pipeline run --all

# Build the network-access cache (cityseer over OS Open Roads, about 15 min)
uv run python stats/oa_network_access.py

# Regenerate every manuscript number, then the figures and the PDFs
uv run python stats/form_size_decomposition.py
uv run python stats/lock_in.py
uv run python stats/access_profile.py
uv run python stats/travel_energy.py
uv run python stats/maup_scale.py
uv run python stats/mixed_use.py
uv run python stats/scenarios.py
uv run python stats/cluster_sensitivity.py
uv run python stats/argument_figures.py
uv run python stats/map_figures.py
latexmk -cd -pdf paper/latex/main.tex
latexmk -cd -pdf paper/latex/extended_data.tex

# Checks
uv run pytest
uv run ruff check .
```

The full recipe, including the manual downloads, is in [REPRODUCTION.md](REPRODUCTION.md).

---

## License

GPL-3.0-only. Author: Gareth TODO.
