"""
Infrastructure module for HFT Research System.

Provides core infrastructure components:
- Configuration management
- Logging and performance tracking
- Metrics computation
"""

from .config import (
    SystemConfig,
    LatencyConfig,
    MarketDataConfig,
    SignalConfig,
    ExecutionConfig,
    RiskConfig,
    BacktestConfig,
    Environment,
    DataSource,
    get_default_config,
    get_backtest_config,
)

from .logging import (
    HFTLogger,
    LogCategory,
    LatencyMeasurement,
    LatencyTracker,
    get_logger,
    get_latency_stats,
    logger,
)

from .performance_metrics import (
    PerformanceTracker,
    Trade,
    Position,
    Side,
    format_metrics_report,
)

__all__ = [
    # Config
    "SystemConfig",
    "LatencyConfig",
    "MarketDataConfig",
    "SignalConfig",
    "ExecutionConfig",
    "RiskConfig",
    "BacktestConfig",
    "Environment",
    "DataSource",
    "get_default_config",
    "get_backtest_config",
    # Logging
    "HFTLogger",
    "LogCategory",
    "LatencyMeasurement",
    "LatencyTracker",
    "get_logger",
    "get_latency_stats",
    "logger",
    # Metrics
    "PerformanceTracker",
    "Trade",
    "Position",
    "Side",
    "format_metrics_report",
]
