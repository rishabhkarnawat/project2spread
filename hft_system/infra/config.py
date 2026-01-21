"""
Configuration Management for HFT Research System
================================================

This module provides centralized configuration management with:
- Type-safe configuration dataclasses
- Environment-specific overrides
- Latency simulation parameters
- Risk thresholds and trading limits

In production HFT systems, configuration is typically:
- Loaded from low-latency key-value stores (Redis, custom solutions)
- Hot-reloadable without system restart
- Version-controlled with audit trails
- Validated against schema before deployment

For this research system, we use static configuration with clear documentation
of what would need to change for production deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import os


class Environment(Enum):
    """Deployment environment - affects logging, latency simulation, etc."""
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    # PRODUCTION would exist in real systems but is out of scope here


class DataSource(Enum):
    """
    Available data sources.
    
    NOTE ON DATA QUALITY HIERARCHY (Critical for recruiter discussions):
    
    1. Co-located exchange feeds (NASDAQ TotalView, CME Market Data)
       - Latency: 10-100 microseconds
       - Full L3 order book, individual order IDs
       - Required for true HFT
    
    2. Consolidated feeds (SIP, CTA/CQS)
       - Latency: 1-10 milliseconds
       - L1/L2 data, NBBO
       - Suitable for slower strategies
    
    3. Commercial data vendors (Bloomberg, Refinitiv)
       - Latency: 10-100 milliseconds
       - Good for research, not execution
    
    4. Free APIs (Yahoo Finance, Alpha Vantage) - WHAT WE USE
       - Latency: 100ms - seconds
       - Delayed quotes, limited granularity
       - Suitable ONLY for research/education
       
    This system abstracts the data source to allow future upgrades.
    """
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    SIMULATED = "simulated"
    # Future: NASDAQ_TOTALVIEW, CME_MARKET_DATA, LOBSTER


@dataclass
class LatencyConfig:
    """
    Latency simulation parameters.
    
    Real HFT latency considerations:
    - Network propagation: ~1 microsecond per 200m of fiber
    - Switch latency: 100-300 nanoseconds (FPGA-based)
    - Kernel bypass: Eliminates ~10 microsecond syscall overhead
    - Memory access: L1 cache ~1ns, L3 ~10ns, RAM ~100ns
    
    We simulate these delays at millisecond granularity for research.
    """
    # Base network latency (simulates distance to exchange)
    base_latency_ms: float = 1.0
    
    # Jitter parameters (network congestion, processing variation)
    jitter_mean_ms: float = 0.1
    jitter_std_ms: float = 0.05
    
    # Latency spikes (rare but impactful events)
    spike_probability: float = 0.001  # 0.1% chance per message
    spike_multiplier: float = 10.0    # 10x base latency during spike
    
    # Processing delays for different components
    signal_compute_latency_ms: float = 0.05
    risk_check_latency_ms: float = 0.01
    order_serialization_latency_ms: float = 0.02


@dataclass
class MarketDataConfig:
    """
    Market data feed configuration.
    
    In production, this would include:
    - Multicast group addresses
    - Feed handler affinity settings
    - Message rate limits
    - Sequence number gap handling
    """
    source: DataSource = DataSource.YAHOO_FINANCE
    
    # Symbols to track
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "SPY"])
    
    # Update frequency (Yahoo Finance rate limits apply)
    poll_interval_seconds: float = 1.0
    
    # Orderbook simulation depth
    orderbook_levels: int = 10
    
    # Simulated tick generation rate (for backtest)
    simulated_ticks_per_second: int = 100
    
    # Alpha Vantage API key (if using that source)
    alpha_vantage_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY")
    )


@dataclass
class SignalConfig:
    """
    Signal generation parameters.
    
    These parameters would typically be:
    - Optimized via backtesting
    - Regime-dependent (different in high/low vol)
    - Updated based on recent performance
    """
    # Microprice signal
    microprice_lookback_ticks: int = 100
    microprice_decay_factor: float = 0.95
    
    # Order flow imbalance
    ofi_window_ticks: int = 50
    ofi_normalization_window: int = 1000
    
    # Latency arbitrage
    correlation_threshold: float = 0.85
    staleness_threshold_ms: float = 50.0
    
    # Meta-signal aggregation weights
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        "microprice": 0.4,
        "order_flow_imbalance": 0.35,
        "latency_arbitrage": 0.25
    })
    
    # Confidence thresholds for trading
    min_confidence_to_trade: float = 0.6
    high_confidence_threshold: float = 0.8


@dataclass
class ExecutionConfig:
    """
    Execution engine parameters.
    
    Real HFT execution considerations:
    - Maker vs taker decisions based on spread and urgency
    - Queue position estimation from historical fill data
    - Venue selection based on rebates and fill rates
    - Iceberg/hidden order detection
    """
    # Slippage model
    base_slippage_bps: float = 0.5  # Basis points
    volatility_slippage_multiplier: float = 2.0
    
    # Fill probability model
    limit_order_fill_probability: float = 0.3
    aggressive_fill_probability: float = 0.95
    
    # Queue position (simplified)
    average_queue_position: float = 0.5  # Middle of queue
    
    # Adverse selection
    adverse_selection_bps: float = 1.0  # Cost of being picked off
    
    # Order sizing
    default_order_size: int = 100
    max_order_size: int = 1000


@dataclass
class RiskConfig:
    """
    Risk management thresholds.
    
    These are CRITICAL in real HFT:
    - Prevent runaway algorithms (Knight Capital lost $440M in 45 minutes)
    - Ensure regulatory compliance (position limits, order rates)
    - Protect firm capital
    
    Risk checks must be:
    - Extremely fast (sub-microsecond in production)
    - Fail-safe (default to blocking if uncertain)
    - Independently monitored
    """
    # Position limits
    max_position_per_symbol: int = 10000  # Shares
    max_gross_position: int = 50000       # Total across all symbols
    max_net_position: int = 25000         # Net exposure
    
    # Loss limits
    max_loss_per_second: float = 1000.0   # Dollars
    max_loss_per_minute: float = 5000.0
    max_daily_loss: float = 50000.0
    
    # Order rate limits
    max_orders_per_second: int = 100
    max_cancels_per_second: int = 200
    
    # Volatility circuit breakers
    volatility_threshold: float = 0.02     # 2% move triggers caution
    extreme_volatility_threshold: float = 0.05  # 5% move halts trading
    
    # Kill switch triggers
    kill_switch_loss_threshold: float = 25000.0
    kill_switch_position_threshold: int = 40000


@dataclass
class BacktestConfig:
    """
    Backtesting engine configuration.
    
    Event-driven backtesting is superior to bar-based because:
    - Captures microstructure effects
    - Realistic fill simulation
    - Tests latency sensitivity
    - Identifies strategy capacity limits
    """
    # Time range
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    
    # Event replay settings
    deterministic_seed: int = 42
    replay_speed_multiplier: float = 1000.0  # 1000x real-time
    
    # Stress test parameters
    latency_spike_scenarios: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 5.0, 10.0]  # Multipliers
    )
    volatility_shock_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 3.0]
    )
    
    # Data dropout simulation
    dropout_probability: float = 0.001
    dropout_duration_ms: float = 100.0


@dataclass
class SystemConfig:
    """
    Top-level system configuration aggregating all components.
    """
    environment: Environment = Environment.RESEARCH
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "hft_system.log"
    
    # Performance profiling
    enable_profiling: bool = True
    profile_output_path: str = "profile_results/"


def get_default_config() -> SystemConfig:
    """
    Returns default configuration for research environment.
    
    In production, this would load from:
    - Configuration management system
    - Environment variables
    - Secure vault for sensitive values
    """
    return SystemConfig()


def get_backtest_config() -> SystemConfig:
    """
    Returns configuration optimized for backtesting.
    """
    config = SystemConfig(environment=Environment.BACKTEST)
    config.latency.base_latency_ms = 0.0  # No artificial delay in backtest
    config.market_data.source = DataSource.SIMULATED
    return config
