"""
Backtesting module for HFT Research System.

Provides event-driven backtesting infrastructure.
"""

from .event_driven_backtester import (
    EventDrivenBacktester,
    BacktestResult,
    SimulationEvent,
    EventType,
    run_quick_backtest,
)

__all__ = [
    "EventDrivenBacktester",
    "BacktestResult",
    "SimulationEvent",
    "EventType",
    "run_quick_backtest",
]
