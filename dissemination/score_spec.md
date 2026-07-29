# NEPI score — specification (v1 draft)

Status: draft for author sign-off. The score is descriptive of the place as-lived; the compositional regression in the paper is for inference and plays no part here. Nothing below introduces a new model: every quantity is already computed per OA in the stats layer.

## Headline: the rate

The headline letter grades the **rate**: everyday amenities reachable within the area's own car catchment, divided by its annual **total** energy per household (kWh, home + car travel). It is a measured ratio per OA, so there are no composite weights to defend.

- Numerator: network amenity count at the OA's own NTS catchment (miles/person ÷ 370 trips/yr, clipped 1.6–25.6 km), read from the cached distance curve (`oa_network_access.parquet`).
- Denominator: metered home energy (gas + electricity) plus NTS-anchored car-travel energy, kWh per household per year.

**Amendment (2026-07-29, author):** the certificate's denominator is total energy, not car-travel energy. The paper's headline rate (access per kWh of driving, 3.9×) is unchanged as a research finding; the certificate divides by total energy so that every technology lever (fabric, heat pump, EV) moves the grade, which the Atlas walkthrough showed is what users expect of a certificate. Bands re-frozen as `NEPI-2021 v2` on the new distribution.

## Bands

- A–G, cut as **household-weighted septiles of the 2021 national distribution of log(rate)**. Equal shares of households, not of areas. A is the top septile.
- Thresholds are computed once, published in the spec, then **frozen** (versioned `NEPI-2021`). Re-scores against frozen thresholds show real movement.
- The two axes beneath the letter are banded the same way (household-weighted 2021 septiles, frozen): energy on log total energy per dwelling (heat + travel), access on log on-foot amenity count (network 1,600 m).
- **Display decision (2026-07-29, author):** only the rate carries the A–G letters and the certificate colours. Energy and access are shown as measured quantities on their own sequential colour scales (the same frozen septile cuts label the scales). Grading the axes invited misreading them as performance scores; energy and access letters remain in the parquet but are not displayed. This also removes the energy-letter saturation issue from the UI (§Label below).

## The label (EPC-style card)

1. Headline letter: the rate band, current.
2. Second letter: the rate band under **full technology deployment** (fabric + heat pump + EV at 100%, the `scenarios.py` transforms).
3. Beneath: the two axes with their own current/potential letters and the underlying numbers (kWh per dwelling split heat/travel; amenities on foot and at catchment).

How the potential letters behave (first full run, 2026-07-29): against the frozen as-lived bands, full deployment lifts nearly every area's energy letter (99% of households reach A on delivered energy) and most rate letters (52% reach A), while the **access letter never moves**. That asymmetry is the card's message, and it is the paper's Discussion made visual: a consumption assessment shows the national programme succeeding everywhere and loses discrimination in the process; access is where lock-in shows, and the rate sits between. An alternative potential definition (fabric + EV only, the `lock_in.py` bound, which would keep the energy letters spread) was considered and rejected: the certificate's potential should reflect the full published pathway, and the access row carries the lock-in.

## Potential computation (deterministic)

Per OA, from `scenario_energy`: gas × EPC fabric factor × boiler_eff/COP (electricity for heat), metered electricity unchanged, travel miles unchanged at EV fleet intensity. Constants as in `scenarios.py` (COP 2.8, boiler 0.90). Access is unchanged by construction, so only the denominator of the rate moves.

## Units

Energy is per dwelling, as metered and billed, matching the paper's canonical specification (household size and floor area are regression controls there, never denominators; the household-size elasticity is about 0.5, so per-person normalisation is rejected). The score has **one mode**: as-lived per dwelling. No per-person or equal-household-size toggle; the OA card shows household size and floor area as context lines, and the site's about page carries the one-paragraph rationale. The equal-household-size counterfactual remains a paper-level result.

## Coverage and edge cases

- OAs with missing or low-coverage inputs (no gas meters, sparse EPC, suppressed energy) get a letter with a **low-confidence flag**, not an exclusion, using the same coverage rules as the stats layer. The flag and the reason are shown on the card.
- Catchment clipping (1.6–25.6 km) and the 370 trips/yr constant are inherited as definitional; the paper's sensitivity sweep (300/440) covers their contestability.

## Non-goals for v1

- No composite index, no weights, no modelled counterfactuals. Modelled form-change what-ifs (densification) are out of scope entirely; XGBoost was dropped from the plan 2026-07-29.
- England only, OA 2021 geography, 2021–22 data vintage as in the paper.

## Deliverable

`stats/nepi_score.py`: reads the same artefacts as `oa_data.py`, writes `statistics/oa_nepi_score.parquet` (OA21CD, rate, letters current/potential for rate and both axes, the per-OA lever inputs, coverage flags). Print summary reports the band thresholds and the national letter distribution. Tests cover threshold freezing and the potential transform.
