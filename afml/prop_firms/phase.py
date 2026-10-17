"""
Trading-phase enumeration for prop-firm evaluation programs.

Split into its own module because both PropFirmRuleSet (which keys
phase_targets by Phase) and PropFirmAccountState (which tracks the
current Phase) depend on it. Keeping Phase here avoids a circular
import between rule_set.py and account_state.py.
"""

from __future__ import annotations

from enum import Enum


class Phase(Enum):
    CHALLENGE_PHASE_1 = "challenge_phase_1"
    CHALLENGE_PHASE_2 = "challenge_phase_2"
    FUNDED = "funded"
