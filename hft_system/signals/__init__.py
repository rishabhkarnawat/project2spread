"""
Signal generation module for HFT Research System.

Provides alpha signals based on market microstructure:
- Microprice signal (order book imbalance)
- Order flow imbalance (OFI)
- Latency arbitrage detection
- Signal aggregation
"""

from .microprice import (
    MicropriceCalculator,
    MicropriceSignal,
)

from .order_flow_imbalance import (
    OrderFlowImbalanceCalculator,
    OFISignal,
)

from .latency_arbitrage import (
    LatencyArbitrageDetector,
    LatencyArbSignal,
    SymbolPair,
)

from .signal_aggregator import (
    SignalAggregator,
    CombinedSignal,
)

__all__ = [
    # Microprice
    "MicropriceCalculator",
    "MicropriceSignal",
    # OFI
    "OrderFlowImbalanceCalculator",
    "OFISignal",
    # Latency arb
    "LatencyArbitrageDetector",
    "LatencyArbSignal",
    "SymbolPair",
    # Aggregator
    "SignalAggregator",
    "CombinedSignal",
]
