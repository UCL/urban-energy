"""
Demand-side scenario specifications.

Two **hypothetical** scenarios for hand-on-the-dial exploration:

    - heat-pump rollout (gas heat → electric heat at COP=3)
    - EV rollout (ICE transport → BEV at efficiency ratio 3.5)

These are not forecasts. They answer "what if X% of homes / vehicles
electrified?" given the per-OA fuel data we already have. They apply
uniformly across OAs (the hypothetical share is the same everywhere)
but the *effect* differs per OA because OAs with more gas exposure
benefit more from heat-pump rollout, etc.

When B5+ (modelled adoption with archetype-conditioned curves) lands,
these uniform-share scenarios become the baseline against which
modelled pathways are compared.
"""

from __future__ import annotations

from dataclasses import dataclass

# Engineering constants (from CIBSE Guide F + DfT EV consultation papers).
HEAT_PUMP_COP: float = 3.0  # heat output per unit electricity input
EV_EFFICIENCY_RATIO: float = 3.5  # BEV is ~3.5× more efficient than ICE per km


@dataclass(frozen=True)
class DemandScenario:
    """
    Demand-side adoption shares.

    `hp_share`  fraction of gas-heated dwellings switched to heat pumps.
    `ev_share`  target absolute share of vehicles that are BEV. If higher
                 than the OA's actual `bev_share`, this overrides it
                 (uniform hypothetical target across all OAs).
    """

    hp_share: float = 0.0
    ev_share: float = 0.0

    def is_neutral(self) -> bool:
        return self.hp_share == 0.0 and self.ev_share == 0.0


# Named scenarios used by the Atlas. Keys correspond to Mode toggle slugs.
NEUTRAL = DemandScenario(0.0, 0.0)
HP_ROLLOUT_80 = DemandScenario(hp_share=0.80, ev_share=0.0)
EV_ROLLOUT_80 = DemandScenario(hp_share=0.0, ev_share=0.80)
