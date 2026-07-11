# Submission checklist — Nature Cities (Phase 0)

Working state as of 2026-07-11. Items marked (user) need an action only the author can take; everything else is prepared in-repo.

## Number wiring (no hand-typed results)

Every model number in `latex/main.tex` and `latex/extended_data.tex` is a `\nepi<Key>` macro from `latex/numbers.tex`, and the results tables are `\input` fragments (`latex/tab_*.tex`). Both are written by the stats scripts through `stats/ledger.py` at the point of computation. To refresh after any analysis change:

```bash
uv run python stats/form_size_decomposition.py
uv run python stats/lock_in.py
uv run python stats/access_profile.py
uv run python stats/travel_energy.py
uv run python stats/maup_scale.py
uv run python stats/scenarios.py
cd paper/latex && latexmk -pdf main.tex extended_data.tex
```

A hand-typed result number in the manuscript is a bug. Static exceptions (kept literal by policy): definitional constants (1,600 m, 370 trips/yr, distance bands), deliberately approximate prose ("about 2,500 miles"), and the Extended Data amenity source counts (acquisition-time facts recorded in `results_snapshot.txt`).

## Done

- **Number wiring live**: every model number in both documents is a `\nepi` macro; the two-axis, scenario, Oster and MAUP tables are ledger-rendered complete tabulars (`latex/tab_*.tex`). The wiring already caught one real transcription error (family-size CI 1.16→1.15).
- **Expert review pass applied** (2026-07-11): editor (37 findings), statistician (13 findings + referee-attack analysis), data-visualisation (25 findings). All must-fix and should-fix items applied: CCC "legislated" overclaim corrected, per-capita/per-person contradiction fixed, alpha-zero floor description made accurate, ED Oster caption corrected (attenuation, not suppression), travel Methods completed (household-size step, band midpoints, intensity provenance), catchment trip-rate rationale added, competition-for-amenities limitation added, MAUP heat attenuation surfaced in main text, plus ~20 prose and figure-craft fixes.
- **Figures**: charts now also emit vector PDFs and the manuscript includes them; the two-panel England map is built at print width; threshold-label and bracket collisions fixed; the in-figure editorial blocks can be suppressed for submission with `NEPI_PLAIN_FIGS=1` (policy decision at submission: Nature figures conventionally carry no in-figure titles).
- **Figure-set overhaul** (2026-07-11, dataviz + editor reviews applied): Fig 1 is an X-Y plot of the four types; the England map is quantile-classed with Greater London insets; the rate figure is a three-panel construction (catchment reach, car energy, rate with bootstrap CI); the forest plot is two panels with the rate row added; the energy bars moved to Extended Data; a new Extended Data equity figure shows the deprivation gradient and its inversion in the strongest markets; the scenario figure's access panel states 27x as text rather than fake bars; figure numbering matches first-citation order. Both energy views (per dwelling and equal household size) appear in Table 1, the forest plot and the scenario figure. Extended Data pruned to non-duplicative items and carries the co-author block. Main set is 6 figures + 1 table; Extended Data is 4 figures + 4 tables.
- GIAS record count verified against the built layer (50,631).

- Manuscript: [latex/main.tex](latex/main.tex) → `main.pdf` (compiles clean, sn-nature reference style, 7 figures + 1 table, Extended Data cross-references wired).
- Extended Data: [latex/extended_data.tex](latex/extended_data.tex) → `extended_data.pdf` (4 figures, 4 tables, numbers matched to [results_snapshot.txt](results_snapshot.txt)).
- Cover letter draft: [cover_letter.md](cover_letter.md) (suggested referees still TODO).
- Citation metadata: `CITATION.cff` and `.zenodo.json` at the repo root, ready for the archived release.
- References: all cited keys resolve, and every cited entry was verified against the published record (2026-07-11). Four entries corrected (winkler2023 and summerfield2019 author lists, firth2024 authors + volume/issue/pages/year, ukgovernment2023 year/publisher); three citations added for previously unreferenced claims (patterson1996 energy productivity, sorrell2007 rebound, alonso1964 access capitalisation); miscast citations fixed (buyuklieva2023 dropped from "metered" claims, crawley2019 re-scoped to measurement error, ewing2017 meta-analysis now cited).
- **Full prose pass applied** (2026-07-11): 132 sentence-level findings across main.tex, extended_data.tex and the cover letter — terminology standardised (rate/ratio, car catchment, home vs household energy, measure vs instrument, median counterpart, British adverbs), long sentences split, elliptical constructions expanded, two cross-document inconsistencies reconciled (the argument rests on access and the rate; MAUP home contrast "attenuates towards parity" with CI spanning one).
- **Computed robustness values wired** (2026-07-11): the Methods trip-rate claim now cites the computed sweep (300/370/440 trips/yr → advantages 1.37/1.26/1.28×, rates 4.2/3.9/3.9×), and the travel section reports the equal-household-size travel gap (2.68×, CI 2.41–2.97) via `\nepitravelFamGap`.
- Co-author review guide: [review_guide.md](review_guide.md) (claims, judgment calls, open decisions, what is needed from each author).
- Affiliation confirmed and filled as "Building Stock Lab, UCL Energy Institute, University College London" in the manuscript, Extended Data and cover letter. Co-authors Steve, Daniel and Matteo are placeholdered in `main.tex`; surnames, emails and any second affiliations still to be supplied (user).

## Remaining, in order

1. (user) Confirm affiliation wording; add acknowledgements (funding, if any) in `main.tex` §Declarations.
2. (user) Code DOI: push a tagged release (e.g. `v1.0.0-submission`), archive it on Zenodo (`.zenodo.json` supplies the metadata), then paste the concept DOI into `main.tex` §Code availability and `CITATION.cff`.
3. (user) Data DOI: deposit the built per-OA artefacts on Zenodo and paste the DOI into `main.tex` §Data availability. Suggested deposit set (all under `$URBAN_ENERGY_DATA_DIR/statistics/`): `oa_energy_consumption.parquet`, `oa_epc.parquet`, `oa_network_access.parquet`, `oa_access.parquet`, `oa_hdd.parquet`, `oa21_ruc21.parquet`, `nts_mileage_by_ruc.parquet`, `lsoa_imd2025.parquet`, `lsoa_vehicles.parquet`, `postcode_oa_lookup.parquet`, plus `paper/results_snapshot.txt`.
4. (user) Suggested referees in the cover letter (2–4 names; the obvious pools are the metered-energy group behind NEED/SERL analyses, the accessibility-instruments literature, and the travel-and-built-environment meta-analysis line).
5. Add the `lineno` documentclass option for the review copy if requested by the portal; rebuild both PDFs.
6. (user) Submit `main.pdf` + `extended_data.pdf` + cover letter via the Nature Cities portal; ORCID at submission.

## Fallback ladder (decided)

Nature Cities → npj Urban Sustainability → Environment & Planning B → Energy Policy. The manuscript transfers to npj Urban Sustainability essentially unchanged.
