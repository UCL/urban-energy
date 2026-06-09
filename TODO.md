# TODO

Methodological and design items requiring deliberation before action.

> Status (2026-06-09): prioritisation now lives in [ROADMAP.md](ROADMAP.md).
> #6 is **resolved** (Form under-recording flags implemented). #4 and #5 are
> **deferred** with the LiDAR/morphology path. #1, #2, #3 remain open analysis
> decisions and need no pipeline re-run.

## 1. Per-household vs per-capita

The composite NEPI is currently published per household per year. Household
size varies systematically with archetype (flats are typically smaller
households than detached dwellings), so per-household values understate
the per-capita energy intensity of compact dwelling types relative to
sprawling ones. Decide whether to:

- Keep per-household as the canonical unit and add per-capita as a Unit
  toggle on the dashboard.
- Switch to per-capita as canonical and demote per-household.
- Publish both side by side in the panel without a toggle.

Resolution depends on the intended interpretation: per household is the
natural unit for billed energy; per capita is the natural unit for
emissions accounting and equity comparisons. The choice has implications
for how the morphology gradient reads.

## 2. Critical interpretation of gradient-booster outputs

The XGBoost models in the What-if workbench predict per-OA NEPI surfaces
from nine planner-controllable features with monotonic constraints. Two
interpretation issues need to be addressed before drawing structural
claims from the model output:

- The marginal contribution attributed to each feature by SHAP is
  conditional on the model's representation of the joint feature
  distribution, not a structural causal effect. Co-linear features
  (density, dwelling type, build year) share explanatory power; SHAP
  partitions it deterministically but the partition is model-specific.
- The reference baseline for "vs actual" deltas in the workbench is the
  mid-archetype interpolation. This is a convenience reference, not a
  causal counterfactual. Document this explicitly and consider whether
  to expose alternative baselines (national median, archetype median).

## 3. Lock-in: audience, definition, robust unpacking

The Lock-in surface is currently defined as the residual demand after the
2050 grid + 100 % heat-pump + 100 % EV adoption. Open questions:

- **Audience.** Who is the surface for? Planners deciding where to build
  read it differently from policy analysts deciding where to retrofit.
  The framing on the dashboard and About page should make the intended
  audience explicit.
- **Definition.** Is "100 % HP + 100 % EV" the right end-state, or should
  the residual be reported under a more realistic ceiling (e.g., 80 / 80,
  the realistic upper bound under current trajectories)? The choice
  changes the absolute residual but not the relative geography.
- **Unpacking.** The residual collapses heating and transport into a
  single composite. Surface-level lock-in (form-only residual,
  mobility-only residual) would expose which demand component drives
  the lock-in for a given OA. Decide whether to add per-pillar lock-in
  views.

## 4. LiDAR vs WALS

> **Deferred (2026-06-09).** The LiDAR/morphology path is deferred from the lean
> rebuild — nothing published or live consumes its columns beyond a single
> `height_mean` table cell. Revisit this source choice only if the morphology
> dimension is reinstated.

Building heights currently come from Environment Agency LiDAR composite
DSMs. Costs of LiDAR: large download volumes, periodic gaps in coverage,
moderate processing. Alternative: WALS (Welsh / Wider Area LiDAR Service)
or comparable 3D building datasets that ship pre-computed heights and
roof geometry. Decide whether the additional fields LiDAR enables
(roof-level analysis, sky view factor inputs) justify keeping it, or
whether WALS-equivalent data is sufficient for the morphology features
the pipeline uses today.

## 5. Sky view factor and shadow

> **Deferred (2026-06-09).** Both ride on the LiDAR DSM, which is deferred.
> Reconsider together with item #4.

Two morphology features not currently in the pipeline:

- **Sky view factor.** The fraction of the upper hemisphere visible from
  a given point at street or roof level. Established proxy for solar
  access, urban heat island intensity, and street-canyon ventilation.
- **Shadow / solar exposure.** Annual hours of direct sunlight on the
  building envelope. Direct relevance to solar PV potential and to
  passive heating gains in the heating season.

Both are computable from the existing LiDAR DSM but are non-trivial
additions to the pipeline. Decide whether to add them, and if so whether
they enter the NEPI composite or sit alongside as auxiliary metrics.

## 6. Aggregation pitfalls in metered energy

> **Resolved in part (2026-06-09).** Flag (a) is implemented:
> `urban_energy.form_bias.compute_form_bias_flags` adds `form_flat_share`,
> `form_gas_meter_coverage`, `form_offgas_flag`, `form_bulkgas_flag`, and
> `form_underrecorded_flag` to the OA dataset (Stage 3). Still open: (b) surface
> these on the Atlas About page and in the paper, and (c) an EPC-based correction
> for the most affected OA classes — tracked in [ROADMAP.md](ROADMAP.md).

The Form surface uses DESNZ postcode-level domestic energy aggregated to
OA via the postcode-to-OA spatial lookup. Two known sources of
distortion:

- **Bulk gas purchasing in flats and blocks.** Where a block of flats
  has a single non-domestic gas meter (commercial heat-network supply,
  district heating, communal boilers) the gas consumption is recorded
  in the non-domestic dataset and is therefore absent from the domestic
  postcode totals. The flats appear in the OA without their associated
  heating load, deflating the apparent Form energy of the OA. Direction
  of bias: compact-flat OAs read as more efficient than they actually
  are.
- **Off-gas-grid OAs.** OAs with no mains gas connection (predominantly
  rural, but also some suburban estates and post-1990 developments)
  rely on heating oil, LPG, biomass, or direct electric heating. Oil
  and LPG are not metered at OA granularity in the DESNZ dataset.
  Direction of bias: off-gas OAs read as having no heating load on the
  Form surface, deflating their apparent energy intensity.

Both issues bias the Form surface downward in specific OA types.
Required: (a) a flag in the OA properties identifying high-risk OAs
(low recorded gas relative to dwelling count, high flat share, off-gas
postcode lookup), (b) explicit mention in the About page and paper, and
(c) consideration of an EPC-based fallback or correction for the most
affected OA classes.
