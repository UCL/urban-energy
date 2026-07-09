# Figure review notes (expert critiques + status)

Three expert reviewers (data-visualisation, graphic design, science editing) critiqued
the figure set on 2026-07-09. Most points were applied in the rebuild (commit e45d1a7);
this file records what was done and what is still open, so the work can be resumed.

The current set is ten figures in `paper/figures/`, built by `stats/argument_figures.py`
(F1–F8) and `stats/map_figures.py` (F9–F10), styled through `stats/figstyle.py`.

## Applied (done in the rebuild)

- **F8 lock-in drawn on both axes** (the top structural fix from all three): a two-panel
  chart, energy gap narrowing on the left, access gap frozen at 27× on the right.
- **F10 two-hue maps**: energy in the warm ramp, access in the green ramp, so the two
  panels no longer read as one flipped blue scale.
- **Colour discipline**: energy = brick red, access = green, dwelling type = slate ramp,
  one "gets worse" amber in F8; a shared editorial frame (kicker, finding title,
  mechanism subtitle, source footer) on every figure.
- **F7 rate**: middle types greyed, the 3.9× comparison bracketed.
- **F5 doorstep**: symlog x so every service shows a real gap; numbered title.
- **F2 country**: hexbin density (no overplot blob or integer banding); linear y-axis.
- **F6 reach**: both reference lines (on foot, 25 km) with the 27× and 11× multiples.
- **F4 decomposition**: proper waterfall showing each control's bite.
- **F9 England**: light land + city labels so the pattern is legible.
- **Named label-on-data collisions fixed**: F1 rules behind the headers, F1 sans caption
  (not serif italic), F2 anchors on leaders with a white halo, F4 rotated "parity"
  removed, F6 horizontal reference labels.
- **Format**: one PNG per figure, all full column width, stale duplicates removed.

## Open — to pick up next

**Editor (narrative / structural):**
- **End the figure sequence on the lock-in figure**, not on a map. In `summary.md` the
  maps (F9, F10) currently come after F8, so the deck trails off on a choropleth. Either
  move the maps earlier (with the F2 "it's national" beat) or to supplementary, so the
  payoff (technology cannot fix it) is the last thing the reader sees.
- **Demote F4 (decomposition) to supplementary / a robustness slot.** It is a rigour
  detour mid-arc, and "only ~17% is pure form" reads as a hedge just where conviction
  should build. If kept in-arc, retitle as a positive finding.
- **Consider an explicit opening concept panel** (old lens: energy burned; new lens:
  access bought) and a closing "so what" (form is fixed for generations, so plan compact;
  retrofit cannot substitute). Lower priority: F1 already hooks well, and F8 can carry the
  "so what" if its title states the action.
- Title tweak worth making: put the numbers in F1's title ("burns 2.1× the energy to reach
  1/27 the amenities").

**Data-viz / designer (craft, not yet applied):**
- **F3 energy gradient**: desaturate the car-travel blue toward slate so the red/blue does
  not read as the default matplotlib clash; and add per-segment value labels (heat vs
  travel) so the 3.1× / 1.6× split is legible off the bars, not just the totals.
- **Rounded bar ends / refined marks** across the bar figures (deferred; cosmetic).
- **Residual minor collisions** to tidy: F8 "today" label sits near the 2.2 x-tick; F2
  "median neighbourhood" leader label may clip at the bottom on the linear axis; F10 the
  source footer runs close to the colourbars.

**Data provenance:** the amenity source/filter table is in `summary.md` (Access section);
keep it in sync if any layer filter changes.
