"""
Risk management module for HFT Research System.

Provides critical risk controls:
- Position limits
- Kill switch
"""

from .position_limits import (
    PositionLimitManager,
    PositionLimitConfig,
    LimitType,
    LimitAction,
    LimitBreach,
)

from .kill_switch import (
    KillSwitch,
    KillSwitchEvent,
    KillSwitchReason,
    TradingState,
)

__all__ = [
    # Position limits
    "PositionLimitManager",
    "PositionLimitConfig",
    "LimitType",
    "LimitAction",
    "LimitBreach",
    # Kill switch
    "KillSwitch",
    "KillSwitchEvent",
    "KillSwitchReason",
    "TradingState",
]
