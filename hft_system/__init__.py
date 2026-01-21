"""
HFT Research System
====================

A production-grade High-Frequency Trading research and simulation system.

This system demonstrates:
- Event-driven architecture
- Market microstructure understanding
- Quantitative rigor
- Latency-aware design
- Risk management best practices

MODULES:
- data: Market data feeds and order book simulation
- signals: Alpha signal generation (microprice, OFI, latency arb)
- execution: Smart order routing and execution simulation
- risk: Position limits and kill switch
- backtest: Event-driven backtesting engine
- infra: Configuration, logging, and metrics

NOTE: This system uses free data sources (Yahoo Finance) which are
NOT suitable for real HFT. See README.md for details on data quality
and how to adapt for production use.
"""

__version__ = "1.0.0"
__author__ = "HFT Research Team"

from .infra import (
    SystemConfig,
    get_default_config,
    get_backtest_config,
    logger,
    PerformanceTracker,
)

from .data import (
    create_data_feed,
    SimulatedDataFeed,
    OrderBook,
    OrderBookSimulator,
)

from .signals import (
    SignalAggregator,
    CombinedSignal,
    MicropriceCalculator,
    OrderFlowImbalanceCalculator,
    LatencyArbitrageDetector,
)

from .execution import (
    SmartOrderRouter,
    ExecutionSimulator,
    Order,
    OrderType,
)

from .risk import (
    PositionLimitManager,
    KillSwitch,
    TradingState,
)

from .backtest import (
    EventDrivenBacktester,
    BacktestResult,
    run_quick_backtest,
)

__all__ = [
    # Config
    "SystemConfig",
    "get_default_config",
    "get_backtest_config",
    # Logging
    "logger",
    # Data
    "create_data_feed",
    "SimulatedDataFeed",
    "OrderBook",
    "OrderBookSimulator",
    # Signals
    "SignalAggregator",
    "CombinedSignal",
    "MicropriceCalculator",
    "OrderFlowImbalanceCalculator",
    "LatencyArbitrageDetector",
    # Execution
    "SmartOrderRouter",
    "ExecutionSimulator",
    "Order",
    "OrderType",
    # Risk
    "PositionLimitManager",
    "KillSwitch",
    "TradingState",
    # Backtest
    "EventDrivenBacktester",
    "BacktestResult",
    "run_quick_backtest",
    # Metrics
    "PerformanceTracker",
]
