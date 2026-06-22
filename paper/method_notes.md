# Method notes & literature to incorporate

A running ledger of the methodological decisions taken in the analysis, the
literature that supports (or qualifies) them, and what must be folded into
`PAPER.md` and `literature_review.md` when the manuscript is drafted. This is a
checklist of decisions + citations, not paper prose. Keep it current as the
analysis evolves.

Canonical numbers live in [`summary.md`](summary.md); reproduce via the `stats/`
scripts named under each decision.

---

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

## Open items still to resolve before the paper

1. **`PAPER.md` is the DEFERRED old-method draft** — three-surface / coverage /
   XGBoost analysis with dominant-type numbers (1.46×, 2.00×, A–G scorecard). It is
   *not* a stale copy of the current two-axis numbers and was deliberately NOT
   renumbered (its surrounding prose is about the old method). The canonical current
   statement is `summary.md`. If/when the paper is revived, rebuild it on the
   two-axis basis rather than patching numbers.
