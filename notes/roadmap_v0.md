# Research Roadmap

Development roadmap for strengthening the hypotheses, methods, and analytical framing of this project.

**Status:** Early-stage development
**Last updated:** 2026-02-06

See [TODO.md](TODO.md) for implementation-level tasks.

---

## 1. Hypotheses

The current hypotheses (H1-H3 in [stats/README.md](stats/README.md)) are well-structured and testable. The following areas would sharpen them further.

### 1.1 Theoretical grounding

The three lock-in mechanisms (floor area, envelope, transport) are presented empirically but would benefit from explicit connection to the theoretical literature. Key threads to weave in:

- **Building physics theory** for the envelope hypothesis: the ~17% per shared wall figure is a strong result, but should be derived from or compared against first-principles heat loss calculations (U-values, thermal bridging). This would let the paper distinguish between the geometric effect (surface-area-to-volume) and the insulation effect, strengthening the "technology cannot close the gap" claim.
- **Urban economics** for the floor area hypothesis: why are detached houses larger? Is this a planning constraint (minimum plot sizes), a market equilibrium (land cost vs building cost), or a preference effect? The answer affects policy implications.
- **Transport geography** for the transport hypothesis: Newman & Kenworthy's density-transport relationship is well-established but has known non-linearities. The current quartile-based approach is a reasonable starting point; consider whether the relationship warrants a more nuanced functional form (e.g. log-density).

### 1.2 Hypothesis specificity

The hypotheses currently state expected magnitudes (e.g. "+60% floor area", "+50% intensity"). These are useful benchmarks but it is worth clarifying:

- **Are these pre-specified or estimated?** If estimated from the data, they are findings rather than hypotheses. Consider reframing: state the directional hypothesis as the formal test, and report the magnitude as a result.
- **Null hypothesis framing.** Each H could be stated as a testable null (e.g. "H1₀: Mean floor area does not differ by built form after controlling for construction era") to make the statistical testing more explicit.

### 1.3 The "persistence" claim

The combined lock-in table shows +50% current vs +51% best-tech, which is the paper's central result. To make this robust:

- **Define "best tech" precisely.** The current scenario uses Passivhaus + EV. Spell out the assumptions: what U-values, what vehicle efficiency, what energy mix? This makes the claim reproducible and lets reviewers assess whether the scenario is realistic or optimistic.
- **Sensitivity of the ratio.** The persistence claim is that the _proportional_ penalty is approximately constant. Test this across a range of technology scenarios (not just current vs best) to show it holds as a general principle, not just at two endpoints.
- **Acknowledge where the ratio might break.** Heat pumps, for instance, change the heating fuel entirely — do the envelope penalties look different for gas vs electric vs heat pump systems? This doesn't weaken the paper; it shows analytical depth.

---

## 2. Methods

### 2.1 Matched comparison design

The matched comparison (1945-1979, 80-100 m2) is the core method and is well-motivated. Areas to develop:

- **Sample size in the matched window.** The current matched sample has 151 detached vs 2,450 semi-detached vs 3,443 mid-terrace. The detached count of 151 is small for the study's key comparison. Consider:
  - Widening the matching window (e.g. 70-120 m2) and testing sensitivity of results to the window width.
  - Reporting confidence intervals prominently — with n=151 the uncertainty on the detached mean is non-trivial.
  - Using propensity score matching or coarsened exact matching as a complementary approach that could retain more observations.
- **Multiple matching windows.** Running the matched comparison across several era bands (pre-1945, 1945-1979, 1980-2000, post-2000) would show whether the envelope penalty is stable over time or has been narrowing with improved building regulations. This is a natural extension that strengthens the argument.
- **Floor area as both a matching variable and H1 outcome.** Currently H1 tests floor area differences and H2 matches on floor area. Be explicit in the paper about why floor area is a confounder for H2 but an outcome for H1 — this is methodologically sound but readers will need the logic spelled out.

### 2.2 Transport energy estimation

The transport hypothesis uses Census car ownership as a proxy, with 12,000 km/year per car assumed. This is the weakest link in the current methods — not fatally so, but it warrants attention:

- **Source the mileage assumption.** DfT National Travel Survey publishes average annual mileage by area type. Using the published figure (and its uncertainty) rather than a round number strengthens credibility.
- **Mileage varies by density.** Households in low-density areas drive more km per car _and_ own more cars. Using a single km/year figure for all density levels understates the transport penalty. Consider density-stratified mileage if DfT data permits.
- **COVID-19 in Census 2021.** The 31% WFH rate is already noted in the limitations. For robustness, consider supplementing with pre-COVID car ownership data (Census 2011) to check whether density-car ownership gradients have shifted.
- **Energy conversion.** The current "kWh-eq" conversion needs documenting: what fuel efficiency, what kWh per litre? Similarly, the EV scenario needs explicit assumptions (kWh per km, grid carbon intensity).

### 2.3 Mediation analysis

The Baron-Kenny approach for decomposing density -> building type -> energy is a reasonable starting point. As the project matures:

- **Bootstrap confidence intervals.** The standard Baron-Kenny method relies on the Sobel test which assumes normality of the indirect effect. Bootstrapping the indirect effect (Preacher & Hayes, 2008) is now standard practice and straightforward to implement.
- **Clarify the mediation question.** Is the argument that density _causes_ building type composition, which _causes_ energy differences? Or that density and building type are jointly determined by planning decisions? The mediation framing implies a causal chain — make sure the theoretical model supports this.

### 2.4 Spatial structure

The project already has spatial regression infrastructure (Moran's I, cluster-robust SEs). For the next stage:

- **Decide whether spatial models are core or supplementary.** With geographically clustered EPCs from a single conurbation, spatial autocorrelation is almost certain. If the Moran's I test is significant, the non-spatial results need at minimum a robustness check showing that conclusions hold under spatial error correction.
- **Scale of analysis.** The current design links individual EPCs to OA/LSOA Census data. Be explicit about the modifiable areal unit problem (MAUP) — would results change at a different aggregation level?

---

## 3. Scope and framing

### 3.1 Study area

The full analysis targets England. Greater Manchester (173,907 EPCs) is the development and testing case — large enough to exercise the full pipeline, with a range of urban forms from dense inner city to suburban fringe.

Once the methods are stable on the GM subset, the pipeline will be run against the national EPC dataset (~30M certificates). Points to address for the national rollout:

- **Climate variation.** England spans ~5 latitude degrees and significant heating degree-day variation. The envelope penalty may differ between the North East and the South West. Including HDD as a control (or stratifying by climate zone) becomes important at national scale.
- **Regional housing stock differences.** Terraced housing dominates in Northern mill towns; detached estates are more common in the South East. The national analysis will have better statistical power for all built form categories but should check for regional heterogeneity in the lock-in magnitudes.
- **Computational scale.** Processing ~30M EPCs with spatial joins, matching, and spatial regression will require chunked processing or sampling strategies. Worth designing the pipeline with this in mind now.
- **Matched sample sizes.** The current GM matched comparison has only 151 detached homes (1945-1979, 80-100 m2). At national scale this will grow substantially, resolving the current power limitation. The matching design can then be reassessed — more observations allow tighter matching windows or additional matching variables.

READMEs should adopt consistent language, e.g. "Development case: Greater Manchester. Target scope: England."

### 3.2 Contribution positioning

The paper's niche — quantifying _structural_ penalties that persist under technology change — is distinctive. To position this clearly:

- **Distinguish from energy efficiency literature.** Most EPC-based studies focus on improving efficiency ratings. This paper asks a different question: given two buildings of equal efficiency, how much does _form_ matter?
- **Distinguish from urban transport literature.** The transport component complements the building physics analysis but is estimated rather than measured. Be clear about which results are primary (building envelope) and which are supplementary (transport).
- **Policy hook.** The persistence claim has direct planning implications (new housing layout decisions lock in energy penalties for decades). Making this connection explicit early in the paper gives reviewers and readers a reason to care.

---

## 4. Documentation (as methods develop)

These are not urgent but will become important as the analysis stabilises:

- [ ] Specify the exact EPC download date and record count before filtering
- [ ] Document random seeds for any stochastic processes
- [ ] Record Python version and key package versions used for final results
- [ ] Add a data flow diagram showing the pipeline from raw sources to final results
- [ ] Write a plain-language summary of each hypothesis and finding (useful for abstract drafting)

---

## Next steps

The most productive areas to develop next, roughly in order:

1. **Widen the matched comparison** to multiple era bands and test sensitivity to the matching window — this directly strengthens the core finding and prepares the design for the larger national sample.
2. **Source and document the transport energy assumptions** — the weakest current link.
3. **Write the theoretical framing** connecting the three lock-ins to the established literature — this shapes the paper's introduction and positions the contribution.
4. **Design the pipeline for national scale** — ensure processing, spatial joins, and matching can handle ~30M EPCs efficiently.
5. **Update READMEs** with consistent "GM development case / England target" language.
