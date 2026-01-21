"""
Data module for HFT Research System.

Provides market data infrastructure:
- Market data feeds (real and simulated)
- Order book simulation
- Market impact modeling
"""

from .market_data_feed import (
    MarketDataFeed,
    YahooFinanceDataFeed,
    SimulatedDataFeed,
    MarketDataEvent,
    MarketDataType,
    Quote,
    TradeData,
    create_data_feed,
)

from .orderbook_simulator import (
    OrderBook,
    OrderBookSimulator,
    OrderSide,
    PriceLevel,
    MarketImpactModel,
)

__all__ = [
    # Feeds
    "MarketDataFeed",
    "YahooFinanceDataFeed",
    "SimulatedDataFeed",
    "MarketDataEvent",
    "MarketDataType",
    "Quote",
    "TradeData",
    "create_data_feed",
    # Order book
    "OrderBook",
    "OrderBookSimulator",
    "OrderSide",
    "PriceLevel",
    "MarketImpactModel",
]
