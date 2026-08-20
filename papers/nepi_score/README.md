# The NEPI score and the online Atlas — outline

A methods-and-instrument paper: the two-axis evaluation (energy spent, access gained, and the rate between them) translated into a per-neighbourhood A–G score for England, with the interactive Atlas as the published instrument.

## What exists

- `stats/nepi_score.py` — the A–G score on the rate; band thresholds frozen as NEPI-2021 v2 (`dissemination/nepi_bands_2021.json`).
- `stats/atlas_export.py` → `site/` — aggregates, PMTiles, and postcode shards for the static Atlas.
- `dissemination/` — score specification, Atlas architecture, launch checklist.

## What the paper would add

1. Score construction and rationale: why the rate is the scored quantity, the band-freezing protocol, and stability of the bands under re-estimation.
2. Validation: agreement with the underlying axes, sensitivity to the travel construction, and behaviour at the score boundaries.
3. Distributional reading: how scores fall across regions, deprivation, and tenure, connecting to the equity analysis of the companion paper.
4. The Atlas as an instrument: what a postcode-level score can and cannot support in planning and retrofit decisions.

The companion (lock-in) manuscript in `paper/` supplies the measured axes and the headline gaps; this paper cites them rather than re-deriving them.
