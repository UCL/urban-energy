# The NEPI Argument — canonical statement

**Purpose.** Single source of truth for *what NEPI claims and why*. Everything else expands
or supports it:

| Doc | Role |
|-----|------|
| **`paper/argument.md`** (this file) | The argument: hypothesis + claims + reasoning, distilled |
| [`PAPER.md`](../PAPER.md) | The formal IMRaD expansion |
| [`paper/methodology_review.md`](methodology_review.md) | Adversarial audit — where it's weak |
| [`paper/robustness_plan.md`](robustness_plan.md) | The worklist of fixes |
| [`README.md`](../README.md) | Public pitch + status |

Confidence tags: **[solid]** (verified, multiple methods agree), **[provisional]** (one
method), **[open]** (identified, not yet done). Numbers are current national-OA estimates
(~174k OAs, all from open measured data).

---

## 1. Hypothesis — two axes, and a rate

> Neighbourhood form shapes two things a household experiences: the **energy** it spends —
> to heat the home and to travel — and the **access** it gets to everyday destinations. The
> measure of a place is **not how much energy it consumes, but how much access that energy
> buys** — function per unit energy (the trophic view: a dense neighbourhood passes energy
> through many layers; sprawl dissipates it in one pass).

So the argument has **two measured axes** and a **rate**, not three summed surfaces:

- **⚡ Energy** (kWh/household/year) = **heat** (metered) + **car travel** (anchored to
  measured NTS mileage). What the household spends.
- **🌳 Access** = **nearest distance** to everyday services (measured from the street
  network). What the place gives back.
- **📐 The rate** = access ÷ energy. The headline: compact form delivers more access per kWh.

Each axis must survive the same test: *is the association with form real, or an artefact of
who lives there and what they already own?* The two axes **share a cause** — compact form
drives both — but are kept separate because they differ in **what technology can fix**: energy
is partly optimisable (insulate, electrify), access is structurally locked (§4). That
difference is the lock-in story.

---

## 2. The Energy axis (heat + travel)

Energy is the one thing we can largely **measure**: metered heating, plus car travel
anchored to measured national mileage. Both halves below.

### 2a. Form — heating energy and dwelling form

**Question.** Does low-density form (detached) **intrinsically** use more heating energy than
compact form (flats) — or is the difference just bigger homes and more people?

**Data.** Outcome = **official metered energy** (DESNZ sub-national gas + electricity → OA;
real bills, not modelled). Dwelling type from Census 2021 TS044 ("flat" = purpose-built
blocks only).

**Observation [solid].** Detached neighbourhoods use **~1.5× the energy per household of
flats** (≈ 10,200 → 15,500 kWh/yr); 1.35× per person.

**Why no single "per X" settles it [solid].** Every denominator forces a slope of 1 and
distorts something else. **Per-m² is actively misleading** — energy ∝ floor area^**0.68**, so
kWh/m² mechanically *falls* with size, flattering large dwellings. The question is only
answered by **comparing like-for-like**.

**Same-size test [solid].** Flat areas ≈ 44–83 m², detached ≈ 71–163 m² — overlapping only
near 77 m². Where they overlap (~600 comparable neighbourhoods, holding size *and* household
size): **detached = 1.20×** a flat (N=637, p<10⁻⁵); corroborated by quintile stratification
(1.12–1.17×) and a full-sample decomposition (1.14×).

**Decomposition of the 1.5× [solid].** ~63% is the **entangled** bigger-dwellings-+-more-
people bundle (size and occupancy collinear — can't be split); ~7% age/income; **~30% is the
intrinsic form/fabric penalty ≈ 1.15× (1.12–1.20×)**.

**Causal reading [provisional].** Dwelling size is a *consequence* of low-density form
(mediator); household size is *self-selection* (confound). The stock can't disentangle them —
which is itself the finding: low density, big dwellings, and big households are built together.

**Confounds checked [solid].** Build age robust (1.47–1.50× across specs; detached and flats
same ~1971 vintage). Boundary-straddle OAs de-duplicated.

**Under-recording check [solid].** Flats *are* under-recorded (26% flagged, communal/bulk
gas; coverage 0.81 vs 0.96–0.99); detached are under-recorded the other way (14% off-gas).
Net is modest: well-measured OAs give **1.42×** vs 1.52×. The premium is robust at ~1.4–1.5×.

**Claim.** Total ≈ **1.5×** (well-measured ≈1.4×) — the bill. Intrinsic ≈ **1.15×** — the
fabric penalty holding size/occupancy/age/income constant. The ⅔ between is the inseparable
size+people bundle low-density form co-produces. Per-m² dropped; per-household headline.

### 2b. Travel — car-travel energy by constrained disaggregation

**Question.** How much energy does a household spend on car travel — *all* trips, per place —
not just the commute?

**The undercount [solid].** The old Mobility figure was **commute-only** (journey to work) — a
**~6× undercount** of total car travel. It made travel look like ~8% of household energy when
it is really ~25–40%.

**The data gap.** No open dataset measures *total local vehicle mileage*: the all-trip
origin-destination matrix is commercial (mobile-network, ~£10k+); residence-linked MOT
mileage is access-restricted; open data gives only car *ownership* + *commute* + national
averages.

**The method — constrained disaggregation (open, measured-anchored) [solid].**
- **Anchor (measured):** NTS9904 2024 — car-driver miles/person by **2021 rural-urban class**
  of residence (all-purpose, residence-based; **2,534 urban → 5,217 rural** mi/person).
- **Allocate (measured per-OA):** cars-per-person + commute distance redistribute mileage
  *within* each class.
- **Conserve:** each class's population-weighted mean reproduces the NTS figure **exactly**
  (verified to the integer) — measured total preserved, each OA varies locally, **no
  double-count**.
- **Energy:** × fleet intensity (DVLA `bev_share`, EV vs ICE).

**Result [solid].** Car travel **Flat 3,239 → Detached 9,073 kWh/hh** (≈2.8×); travel is
**24–37%** of household energy.

**Measured vs assumed.** *Measured per place:* car ownership, commute distance, household
size, fleet mix, + the NTS class mileage anchor. *National constants only:* ECUK energy-per-km
and one within-class commute elasticity (0.3).

### Combined energy axis [solid]

| Type | heat | car travel | **total** |
|---|---:|---:|---:|
| Flat | 10,196 | 3,239 | **13,675** |
| Detached | 15,462 | 9,073 | **24,363** |

**Flat→Detached total energy gradient = 1.78×.**

---

## 3. The Access axis (nearest distance)

**Measure.** Access = **nearest network distance** to each everyday service (GP, hospital,
school, food, greenspace, transit). Concrete, measured per place — *no model, no penalty
regression, no reference-coverage assumption*.

**Observation [solid].** Distances rise sharply flat→detached: **GP 633 → 1,530 m, hospital
540 → 1,267, food 269 → 816, school 435 → 790**; bus and greenspace near-universal. A flat
neighbourhood reaches its GP ~2.4× closer than a detached one.

**Claim.** Compact form puts everyday destinations within reach; low-density form pushes them
2–3× further away. This is the *return* side — what a household gets — kept on its own axis,
never converted into energy.

---

## 4. The rate, and what explains it

**The rate [solid].** Energy and access, side by side: a flat neighbourhood spends **0.55×**
the energy of a detached one (13,675 vs 24,363) **and** reaches everything ~2–3× closer.
**Compact form delivers far more access per unit energy.** This is a ratio of two measured
quantities — descriptive, *no model required*.

**What explains it [solid].** Neighbourhood structure — **residential density + dwelling mix** —
explains a large share of *both* axes:
- **Access (distances):** directly structural — compact form puts destinations close.
- **Energy:** density + dwelling mix explain **~46% of total household energy** (R²), via two
  channels: **dwelling mix → heating** (flat-heavy areas use less; R²≈0.23) and **density →
  travel** (compact areas drive far less; R²≈0.33–0.58). *(Network *pattern* alone — meshedness
  — explains little of energy; it is density and dwelling type that matter.)*

So energy and access are **not independent — they are causally linked**: the access deficit
(everything far) is what *forces* the travel energy. **Travel energy is the energy cost of low
access**; heating is the separate, dwelling-driven component.

**Why keep two axes, then?** Not because they have different drivers (they don't) — but because
they differ in **what technology can fix**:
- **Energy is partly tech-optimisable** — insulate the homes, electrify the cars.
- **Access is structurally locked** — no technology moves the GP closer.

This is the **lock-in**, and the rate (access per energy) is what makes it legible: even fully
decarbonised, sprawl delivers less access per Joule, because the access deficit is structural
and permanent without rebuilding.

> Corrected causal claim: compact form drives **both** the access *and* the energy — they are the
> **return** and the **cost** of the same structural cause. The rate matters because the cost can
> be optimised by technology while the return (access) cannot.

---

## 5. Lock-in — why the penalty survives decarbonisation

Structure drives both axes (§4), but technology can reach only one of them — the
carbon/infrastructure **lock-in** (Seto et al. 2016; Unruh 2000): built form fixes energy
demand for decades regardless of technology.

- **Electrification** cuts energy *per mile* (EV ~0.20 vs ICE ~0.58 kWh/vkm) — **not the miles**.
- **Insulation** cuts loss *per m²* — **not** the dwelling's **size or exposed surface**.

**Quantified [solid]** (`stats/lock_in.py`: best-practice fabric — EPC-potential intensity ×
floor area — + full electrification):

| Flat→Detached | Flat | Detached | gap |
|---|---:|---:|---:|
| Energy now | 13,435 | 24,536 | **1.83×** |
| Energy optimised | 9,690 | 14,115 | **1.46×** |

Perfect optimisation closes ~60% of the energy **gap**, but a residual **~1.46×** survives, and it
splits across **both** halves — **heat/size ~2,375 kWh and travel/miles ~2,050 kWh**:

- **Heat lock-in is hard** — at best fabric, detached still uses **1.28×** a flat's heat, driven
  by **size** (≈103 vs 62 m²). Insulation fixes per-m² efficiency, not floor area.
- **Travel lock-in is hard** — electrification preserves the **2.8× mileage ratio exactly**
  (detached drives 2.8× the miles, electric or not).
- **Access lock-in is total** — the Access axis is tech-immune: no technology moves the GP closer.

The pattern is general: **technology optimises per-*unit* efficiency (per-m², per-mile) but not
the structural *quantities* (floor area, miles, distance).** So the residual penalty = bigger
homes (heat) + longer trips (travel) + the **entire** access deficit. This is what the trophic
framing makes legible: even fully decarbonised, sprawl delivers **less function per Joule** — you
can clean the energy, but you cannot make the desert a rainforest without rebuilding it.

---

## 6. Claims ladder (at a glance)

| # | Claim | Status |
|---|-------|--------|
| **Energy — Form (heat)** | | |
| F1 | Detached neighbourhoods use ~1.5× a flat's metered energy/household | **solid** |
| F2 | Per-m² is invalid (energy ∝ floor area^0.68) | **solid** |
| F3 | Same-size intrinsic form penalty ≈ 1.15× (1.12–1.20×) | **solid** |
| F4 | ~63% of the gap is the entangled size+people bundle; ~30% intrinsic | **solid** |
| F5 | Size = form consequence; people = self-selection; inseparable | provisional |
| F6 | Flat under-recording inflates the gap ~0.1×, doesn't overturn it (≈1.4× well-measured) | **solid** |
| **Energy — Travel** | | |
| T1 | Commute-only undercounts total car travel ~6× | **solid** |
| T2 | No open dataset measures total local mileage (OD commercial, MOT restricted) | **solid** |
| T3 | Disaggregation conserves the NTS class marginal exactly | **solid** |
| T4 | Car travel ≈ 2.8× flat→detached; 24–37% of household energy | **solid** |
| E1 | Combined energy (heat+travel) ≈ 1.78× flat→detached | **solid** |
| **Access** | | |
| A1 | Nearest-distance access worsens 2–3× flat→detached | **solid** |
| **Rate + structure** | | |
| R1 | Compact form delivers more access per unit energy (descriptive) | **solid** |
| R2 | Structure (density + dwelling mix) explains ~46% of total energy *and* the access gradient — both axes structural | **solid** |
| R3 | Energy & access are cost & return of one structural cause; differ in tech-optimisability (the lock-in) | provisional |
| **Lock-in** | | |
| L1 | Perfect optimisation leaves ~40% of the energy gap (residual ~1.46×), split heat/size (1.28×) + travel/miles (2.8×) | **solid** |
| L2 | Tech optimises per-unit efficiency, not structural quantities (size, miles); access is 100% tech-immune | **solid** |

---

## 7. Open items — next

- **Lock-in — done** (`stats/lock_in.py`, best-fabric × size + full EV): optimised gradient
  1.83×→**1.46×**, ~40% of the penalty survives, split heat/size (1.28×) + travel/miles (2.8×),
  access 100% locked. *(Minor caveat: optimised heat is EPC-modelled potential × area while
  current is metered — a basis mix that doesn't change the conclusion.)*
- **Rate circularity.** Travel energy is the *cost of low access*, so the rate (access ÷ energy)
  partly contains the inverse of its own numerator. Consider rating against heat + an
  idealised/electrified travel cost, so the rate measures the *structural* return cleanly.
- **Access axis firm-up.** Finalise the nearest-distance access measure and the rate as the
  headline.
- **Formal docs.** Reconcile `PAPER.md` results tables + `CLAUDE.md §4` NEPI-models to the
  two-axis frame (still in the old three-surface numbers).

---

## Appendix — superseded framing (for the record)

Earlier drafts summed three kWh "surfaces" (Form + Mobility + Access penalty) and banded the
total A–G. That cost-stack was abandoned because (a) it inverted the trophic philosophy
(measuring total consumption, not function-per-energy), and (b) the Access penalty was a
regression slice of the same transport variable as Mobility, double-counting it. The two-axis
frame above replaces it: Access is the *return*, measured as distance, never summed into the
energy cost. The old A–G banding and the empirical access-penalty model are retired.
