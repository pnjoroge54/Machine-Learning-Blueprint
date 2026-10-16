"""
afml.prop_firm

Prop-firm-constrained position sizing: a configurable rule set
(PropFirmRuleSet), account-state tracking (PropFirmAccountState), and a
two-stage hybrid sizer (PropFirmAwareSizer) whose risk-sizing math is
invariant to which program's rules are in force.
"""

from .account_state import PropFirmAccountState
from .phase import Phase
from .rule_set import FUNDEDNEXT_STELLAR_2STEP, PropFirmRuleSet
from .sizing import (
    PropFirmAwareSizer,
    WParamCalibrator,
    derisking_factor,
    kelly_payoff_multiplier,
    make_sizer,
    news_window_factor,
    prop_firm_sl_ceiling,
)

__all__ = [
    "FUNDEDNEXT_STELLAR_2STEP",
    "Phase",
    "PropFirmAccountState",
    "PropFirmAwareSizer",
    "PropFirmRuleSet",
    "WParamCalibrator",
    "derisking_factor",
    "kelly_payoff_multiplier",
    "make_sizer",
    "news_window_factor",
    "prop_firm_sl_ceiling",
]
