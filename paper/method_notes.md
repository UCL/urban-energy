# Method notes & literature to incorporate

A running ledger of the methodological decisions taken in the analysis and the
literature that supports (or qualifies) them. The manuscript
(`latex/main.tex`) has incorporated these; the ledger remains the decision
history. This is a checklist of decisions + citations, not paper prose.

Canonical numbers are written by the stats scripts to `latex/numbers.tex` via
`stats/ledger.py`; reproduce via the `stats/` scripts named under each decision.

---

## AUDIT 2026-07-09: amenity layer filtering (the access basket)

The seven on-foot amenity layers were audited for over-inclusion. Four needed
attention; the network cache was rebuilt on the cleaned set (`stats/oa_access.py`
`DEST` and `_SERVICES`).

- **Hospital, dropped.** The NHS source is the ETS file (Trusts and Sites), which
  lists every trust *site* (wards, community clinics, CAMHS units, vaccination
  centres), 34,670 records, not hospitals. It is not a credible walkable amenity.
  Health is represented by GP surgeries (12,664) and community pharmacies (11,268),
  both correctly typed. Removing hospital alone moved the on-foot amenity gap only
  from 9.5× to 9.2× (dominant-type median), so it does not drive the result. On the
  fully cleaned six-service basket the on-foot gap is 27.2× compositional (9.7×
  dominant-median), up from 23.9× on the old basket: cleaning the junk sharpened the
  gap rather than propping it up.
- **Greenspace, filtered.** OS Open Greenspace `greenspace_site` (165,978 GB sites)
  was unfiltered, counting religious grounds (22,460), cemeteries (7,748), golf
  courses, bowling greens and tennis courts. Kept: Public Park Or Garden, Playing
  Field, Play Space, Allotments Or Community Growing Spaces (`_GREEN`).
- **Grocery, kept and relabelled.** Supermarkets/hypermarkets (12,208) plus
  "Retailers - other" (80,863). The latter is a mixed FSA catch-all (convenience
  stores, newsagents, off-licences), but corner-shop food access is real everyday
  access in dense areas, so it is kept and labelled "food retail" rather than
  "supermarkets".
- **Schools, kept as all open GIAS establishments.** Filtered to open (status 1/4)
  in `prepare_gias.py`; the 50,631 open establishments include nurseries, colleges
  and children's centres alongside schools, retained as education access.
- **Food, already clean.** `_FOOD` counts restaurants and cafes, takeaways and pubs;
  hotels and mobile caterers (present in the acquired FSA file) are excluded from
  the access measure.

The access numbers in `summary.md` and `PAPER.md` were regenerated from the rebuilt
six-service cache.

## CORRECTION 2026-06-22 — the rate was 6.3×, is now 3.6×

The headline access-per-kWh rate was wrong. The old `access_profile.py` modelled
the **per-OA ratio** `net_amen / transport` directly (Poisson, income-only) and
reported **6.3×**. That double-counts and does not reconcile with the two axes: it
implied a ~5× energy gap, not the reported 3.07× travel gap, and used income-only
controls while the energy axis uses the full set. The rate is a ratio of two
divisions, so the flat:detached value is the **product of the two reported axes** —
access advantage (catchment amenities, flat:det **1.17×**, income-controlled
Poisson) × energy saving (car-travel energy, det:flat **3.07×**, full-confound
log-OLS) = **3.60×**. Now computed that way in `access_profile.compositional_access`
and `argument_figures.access_per_kwh`; fixed in summary/README/CLAUDE/PAPER and the
figure. Lesson: a number matching the code is not a verified number — the *method*
must reconcile across axes. (Triggered the full-pipeline logic audit, below.)

## AUDIT 2026-06-22 — independent logic check of the whole pipeline

Three independent agents traced every computation. Result: the rate was the only
actual error; one scale inconsistency was found and fixed; the rest reconciles.

- **Confirmed sound:** travel energy (units; the NTS class-mean constraint actually
  holds; no double-count of household size), the compositional pure-type model and
  its invariance, the lock-in fabric ratio and EV substitution (no modelled/metered
  mixing), the access counts (catchment radius, jobs weighted by job-count, people
  by population, monotonicity), and the corrected rate.
- **Fixed — mediation/survival scale.** Mediation % and "% surviving" were on the
  `(ratio−1)` excess scale, not the log model's native log scale. Switched both to
  log points: heat mediation **71%→66%** (compositional; 60% regression), lock-in
  "survives" **45%→55%**. Qualitative claims unchanged.
- **Fixed — Oster δ\* guard.** δ\* is only interpretable when confounds *attenuate*
  the coefficient; now reported `n/a` otherwise. The reported rows (heat 1.30→1.02,
  total 1.87→1.28) both attenuate, so δ\* = 0.30 / 1.14 stand.
- **Fixed — stale 6.3× docstrings** in `oa_network_access.py` and `access_profile.py`.
- **Disclosed (not a bug) — heat DV denominator.** Energy ÷ census households with
  gas coverage < 1 does NOT inflate the headline: well-measured areas (coverage ≥
  0.9) give 1.61× vs the 1.60× headline. Noted in PAPER §3.3.
- **Noted — rate coupling.** The catchment radius and the car energy are both
  functions of mileage; this is the intended reading (access reached vs energy to
  reach it), stated in PAPER §3.5.

---

## Decision 1 — functional unit: per dwelling, with family size as a *free control*

**What we do.** Energy is modelled per dwelling. Household size and floor area
enter the regression as covariates with freely estimated coefficients, never as
denominators. Per-person and per-m² are retained only as descriptive lenses.

**Why.** A fixed denominator silently fixes an elasticity the data reject.
Heating is a property of the building envelope, so energy is *sub-linear* in
occupants: the estimated household-size elasticity of heat here is **γ ≈ 0.47–0.54**
(`stats/form_size_decomposition.py`, the ladder's γ line). Per-person division
forces γ = 1, mechanically crediting detached homes for the larger households that
self-select into them; per-m² forces an area-elasticity of 1 (the measured
floor-area elasticity is ~0.2–0.54, so per-m² flatters large dwellings).

**Effect on the numbers.** Heat Det:Flat: per person ≈ parity (the artefact) →
per dwelling **1.60×**; family-size-held **1.27×**; size-held direct **1.17×**
(γ-correct). Total energy **2.12×** per dwelling.

**Literature — agreement (cite in Methods + Discussion):**
- **Huebner & Shipworth (2017)** `huebner2017` — home size per capita is the
  single strongest predictor of per-capita energy; household size is *negatively*
  associated with per-capita demand (economies of scale; sub-linear). Direct
  support for γ < 1 and for rejecting per-person as an inferential unit.
- **Druckman & Jackson (2008)** `druckman2008` — same sub-linearity in the UK
  stock (per-capita energy falls with household size).
- **Norman, MacLean & Kennedy (2006)** `norman2006` — low-density uses 2.0–2.5×
  energy per capita; per m² the advantage narrows to **1.0–1.5× but does not
  reverse**. Use this to state explicitly that **no literature supports "detached
  is more efficient"**; the per-m²/per-person "parity" is a known normalisation
  artefact, not fabric efficiency.

**To incorporate:** a "choice of functional unit" subsection in Methods (the
`## Counting` section of `summary.md` is the seed), and a Discussion sentence
placing the result *with* Norman/Huebner, against naïve per-capita readings.

---

## Decision 2 — metered energy, not EPC/SAP-modelled

**What we do.** The energy DV is DESNZ metered gas + electricity. EPC appears only
in `lock_in.py` as the *potential/current* fabric-improvement ratio, where the
performance gap cancels (both terms modelled).

**Why.** SAP/EPC over-predicts consumption, and the over-prediction is *largest
for the biggest, least efficient (detached, high-S/V, oldest) dwellings* — exactly
the stock whose form penalty we are estimating. An EPC-based DV would inflate the
sprawl penalty with model bias. Our own data show this: EPC-modelled demand
Det:Flat ≈ 1.83× vs metered ≈ 1.44–1.63×; actual/modelled is **0.71 (detached) vs
0.83 (flat)** — detached over-predicted more.

**Literature — agreement (already in `literature_review.md` §2.7; ensure cited in
Methods rationale):**
- **Few et al. (2023)** `few2023` — UK SERL; EPC over-predicts by ~−66 kWh/m²/yr;
  metered between-band gradient <10% of modelled; persists even in SAP-assumption-
  matching homes ⇒ RdSAP is structurally biased, not just behaviour.
- **Firth et al. (2024)** `firth2024` — gap grows 3.6 pp per 1,000 kWh predicted;
  varies by built form; *explicit warning* that morphology–energy associations on
  SAP data may be model artefact. Quote this as the core justification.
- **Crawley et al. (2019)** `crawley2019` — EPC measurement error largest for
  inefficient dwellings.
- **Sunikka-Blank & Galvin (2012)** `sunikka-blank2012` — the prebound effect;
  gap widens with modelled demand.
- **Summerfield et al.** (NEED vs Cambridge Housing Model) — metered gas across all
  EPC bands ≈ band C; CHM over-predicts most for large pre-1930 detached. *Needs a
  proper `references.bib` entry (year/venue to confirm) before citing.*

---

## Decision 3 — deprivation control: overall IMD + income domain

**What we do.** Both `imd_overall_score` and `imd_income_score` (IoD25) are held
as confounds (`_deprivation_cols` in `form_size_decomposition.py`). Collinearity
between them is harmless — it inflates only their own SEs, not the form coefficient.

**Note:** England-only. Wales has no directly comparable IMD, so it stays out
until a harmonised source is added (already flagged in `summary.md`).

---

## Decision 4 — climate confound now included

HadUK-Grid 1 km `tas` (1991–2020) → annual HDD per OA (`data/process_climate.py`
→ `oa_hdd.parquet`), held in every energy ladder. Colder northern/rural siting was
part of the raw form gap; it is now netted out of the direct term.

---

## Resolved (2026-06-20)

- **Gas-coverage robustness** — regenerated under the current model. Coverage
  (gas meters / households) **0.81 flat vs 0.94 detached**; holding gas coverage
  equal the heat gap is **1.42×**; on well-measured areas (coverage ≥ 0.9) it is
  **1.61×** ≈ the 1.60× headline. The communal-heating undercount does not drive
  the result; if anything it slightly understates it. `summary.md` Heat updated.
  (`/tmp/gas_robust.py` is the throwaway; fold into `form_size_decomposition.py` as
  a permanent check if a referee asks.)
- **Access axis control** — *kept* income-only by decision, not oversight: the
  overall IMD's geographic-barriers and living-environment sub-domains are
  themselves access measures, so controlling for them would absorb the effect under
  study. Documented in `access_profile.compositional_access` and
  `argument_figures.py`. Energy axes use overall IMD + income; access uses income.
- **Argument figures** — regenerated on the per-dwelling basis: energy_gradient
  **2.12×**, access_per_kwh **3.6×** (corrected from 6.3×, see above), access_curve **24× → 10×**. `argument_figures.py`
  energy confounds now match the ladder (`_deprivation_cols` + `_hdd_cols`); access
  figures stay income-only.
  *(Superseded 2026-07-09: the amenity-basket clean moved these to 3.9× and 27× → 11×;
  the current numbers are in `summary.md` and `results_snapshot.txt`. The 3.6×/24× here
  are the pre-clean values, kept only as ledger history.)*
- **Self-selection** — handled three ways (`form_size_decomposition.py` §6 +
  `summary.md` "Self-selection"). (a) Access is location-intrinsic ⇒ immune by
  construction. (b) NS-SeC (occupational class, now in the loader as
  `pct_nssec_higher`) added on top of deprivation moves the gap by ~nothing.
  (c) Oster (2019) δ* on a continuous detached-share gradient: **total energy is
  robust (δ* ≈ 1.1)** — much of it is the structural travel gap; **heat alone is
  more confound-entangled (δ* ≈ 0.3)** — its non-flat contrast is largely
  deprivation/tenure. Honest takeaway recorded in the doc: the case rests on
  **total energy + access**, not the heat number alone, and the estimand is
  *place-level*, not a household treatment effect. NOTE: the formal Oster is
  spec-sensitive here (binary dominant-Flat/Detached is collinear at the extremes;
  the no-intercept compositional R² is uncentred) — the continuous-gradient
  intercept-OLS is the defensible vehicle. Mover-based panel (UKHLS) is the
  definitive future test; deliberately not pursued (Gareth: out of scope).

- **Relation to prior work** — `literature_review.md` §6 ("Relation to Prior
  Work: Consistency of the Two-Axis Results") added 2026-06-22: building-energy
  axis consistent with NEED/EHS and Wyatt (2013) / Buyuklieva et al. (2023);
  metered-vs-modelled and the functional-unit lesson reproduce Rode/Norman/
  Summerfield/Few/Firth; travel ~3.1× is the same direction as Echenique et al.
  (2012)'s "modest" marginal effect but an extremes contrast; access axis extends
  Ewing (2018) / Elldér et al. (2022); the access-per-energy rate is the novel
  bit. New refs added: wyatt2013, buyuklieva2023, echenique2012, ellder2022,
  cao2009, summerfield2019. (Summerfield was the old open item — now done.)

## Resolution (historical)

1. The markdown draft was rebuilt on the two-axis basis and later superseded by the
   LaTeX manuscript (`paper/latex/main.tex`), which carries the canonical numbers via
   the ledger. The old three-surface / coverage / XGBoost draft (dominant-type numbers
   1.46×, 2.00×, A–G scorecard) is in git history (formerly
   `paper/archive/PAPER_three_surface_deferred.md`). This file is a decision ledger;
   entries above record the state at their date.
