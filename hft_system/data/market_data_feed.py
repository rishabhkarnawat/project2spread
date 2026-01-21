"""
Market Data Feed Handler for HFT Research System
=================================================

This module provides a unified interface for market data ingestion:
- Real-time data fetching from free APIs (Yahoo Finance, Alpha Vantage)
- Abstract interface for easy replacement with production feeds
- Latency simulation for realistic backtesting
- Event-driven data distribution

CRITICAL DISCLAIMER - DATA QUALITY FOR HFT:
============================================

This system uses FREE data sources which are NOT suitable for real HFT:

┌─────────────────────┬────────────────────┬────────────────────────────────┐
│ Data Source         │ Latency            │ Use Case                       │
├─────────────────────┼────────────────────┼────────────────────────────────┤
│ Co-located feeds    │ 1-100 μs           │ True HFT, market making        │
│ Direct exchange     │ 100 μs - 1 ms      │ Latency-sensitive strategies   │
│ Consolidated (SIP)  │ 1-10 ms            │ Institutional trading          │
│ Bloomberg/Refinitiv │ 10-100 ms          │ Research, slower strategies    │
│ Yahoo Finance ←     │ 100 ms - seconds   │ Research ONLY                  │
└─────────────────────┴────────────────────┴────────────────────────────────┘

WHAT WE'RE MISSING WITH FREE DATA:
- Full order book depth (L2/L3 data)
- Individual order IDs and queue position
- Microsecond timestamps
- Tick-by-tick execution data
- Hidden/iceberg order inference

HOW THIS ARCHITECTURE ADAPTS TO REAL FEEDS:
1. Replace YahooFinanceDataFeed with NasdaqTotalViewFeed
2. Change from polling to multicast UDP receive
3. Add kernel bypass (DPDK/Solarflare) for sub-μs latency
4. Implement hardware timestamping
5. Add feed handler affinity to dedicated CPU cores

The abstraction layer (MarketDataFeed) remains the same.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Awaitable
from collections import defaultdict
import threading

# Import yfinance for real data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

import numpy as np

from ..infra.config import MarketDataConfig, LatencyConfig, DataSource
from ..infra.logging import get_logger, LogCategory


logger = get_logger()


class MarketDataType(Enum):
    """Types of market data events."""
    QUOTE = "quote"           # Bid/ask update
    TRADE = "trade"           # Executed trade
    ORDERBOOK = "orderbook"   # Full orderbook snapshot
    IMBALANCE = "imbalance"   # Auction imbalance
    STATUS = "status"         # Trading status change


@dataclass
class Quote:
    """
    Level 1 quote data (best bid/offer).
    
    In production HFT, quotes include:
    - Microsecond exchange timestamps
    - Sequence numbers for gap detection
    - Venue identifier
    - Condition codes (regular, odd lot, etc.)
    """
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    timestamp: datetime
    sequence_number: int = 0
    
    # Simulated fields for latency modeling
    exchange_timestamp: Optional[datetime] = None  # When exchange sent
    received_timestamp: Optional[datetime] = None  # When we received
    
    @property
    def mid_price(self) -> float:
        """Mid-point price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute terms."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        if self.mid_price == 0:
            return 0.0
        return self.spread / self.mid_price * 10000
    
    @property
    def imbalance(self) -> float:
        """
        Quote imbalance: positive means more bid pressure.
        
        This is a simplified microstructure signal.
        Values range from -1 (all ask) to +1 (all bid).
        """
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid_price": self.bid_price,
            "bid_size": self.bid_size,
            "ask_price": self.ask_price,
            "ask_size": self.ask_size,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "imbalance": self.imbalance,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeData:
    """
    Trade (execution) data.
    
    In real feeds, trades include:
    - Trade ID for deduplication
    - Aggressor side (who crossed the spread)
    - Trade condition (regular, odd lot, late, etc.)
    """
    symbol: str
    price: float
    size: int
    timestamp: datetime
    trade_id: str = ""
    aggressor_side: Optional[str] = None  # "BUY" or "SELL"
    
    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class MarketDataEvent:
    """
    Generic market data event wrapper.
    
    This is the unit of data flowing through the system.
    Event-driven architecture processes these sequentially.
    """
    event_type: MarketDataType
    symbol: str
    timestamp: datetime
    data: Any  # Quote, TradeData, or OrderBook
    sequence_number: int = 0
    
    # Latency tracking
    simulated_latency_ms: float = 0.0


# Type alias for event callbacks
EventCallback = Callable[[MarketDataEvent], Awaitable[None]]


class MarketDataFeed(ABC):
    """
    Abstract base class for market data feeds.
    
    This abstraction allows swapping data sources:
    - Research: Yahoo Finance (free, delayed)
    - Development: Simulated data (fast, deterministic)
    - Production: Exchange feeds (low-latency, expensive)
    
    The interface remains consistent across all sources.
    """
    
    def __init__(self, config: MarketDataConfig, latency_config: LatencyConfig):
        self._config = config
        self._latency_config = latency_config
        self._running = False
        self._subscribers: Dict[str, List[EventCallback]] = defaultdict(list)
        self._sequence_numbers: Dict[str, int] = defaultdict(int)
        self._latest_quotes: Dict[str, Quote] = {}
        self._lock = threading.Lock()
    
    @abstractmethod
    async def start(self) -> None:
        """Start the data feed."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the data feed."""
        pass
    
    def subscribe(self, symbol: str, callback: EventCallback) -> None:
        """
        Subscribe to market data for a symbol.
        
        In production, subscription involves:
        - Sending subscription message to exchange
        - Waiting for snapshot (initial state)
        - Processing incremental updates
        """
        self._subscribers[symbol].append(callback)
        logger.info(
            f"Subscribed to {symbol}",
            category=LogCategory.MARKET_DATA,
            symbol=symbol
        )
    
    def unsubscribe(self, symbol: str, callback: EventCallback) -> None:
        """Unsubscribe from market data."""
        if callback in self._subscribers[symbol]:
            self._subscribers[symbol].remove(callback)
    
    async def _publish_event(self, event: MarketDataEvent) -> None:
        """
        Publish event to all subscribers.
        
        In production, this would be lock-free using:
        - SPSC (single producer single consumer) queues
        - Memory-mapped ring buffers
        - CPU core affinity for deterministic latency
        """
        # Simulate network latency
        if self._latency_config.base_latency_ms > 0:
            latency = self._simulate_latency()
            event.simulated_latency_ms = latency
            await asyncio.sleep(latency / 1000.0)
        
        # Update sequence number
        event.sequence_number = self._get_next_sequence(event.symbol)
        
        # Notify subscribers
        for callback in self._subscribers.get(event.symbol, []):
            try:
                await callback(event)
            except Exception as e:
                logger.error(
                    f"Error in subscriber callback: {e}",
                    category=LogCategory.MARKET_DATA,
                    symbol=event.symbol
                )
    
    def _simulate_latency(self) -> float:
        """
        Simulate realistic network latency with jitter and spikes.
        
        Real network latency has:
        - Base propagation delay (speed of light)
        - Queuing delay (network congestion)
        - Processing delay (switches, NICs)
        - Occasional spikes (GC, context switches)
        """
        base = self._latency_config.base_latency_ms
        jitter = random.gauss(
            self._latency_config.jitter_mean_ms,
            self._latency_config.jitter_std_ms
        )
        
        # Occasional latency spikes
        if random.random() < self._latency_config.spike_probability:
            spike = base * self._latency_config.spike_multiplier
            return max(0, base + jitter + spike)
        
        return max(0, base + jitter)
    
    def _get_next_sequence(self, symbol: str) -> int:
        """Get next sequence number for a symbol."""
        with self._lock:
            self._sequence_numbers[symbol] += 1
            return self._sequence_numbers[symbol]
    
    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get the most recent quote for a symbol."""
        with self._lock:
            return self._latest_quotes.get(symbol)
    
    def get_all_latest_quotes(self) -> Dict[str, Quote]:
        """Get all latest quotes."""
        with self._lock:
            return dict(self._latest_quotes)


class YahooFinanceDataFeed(MarketDataFeed):
    """
    Market data feed using Yahoo Finance.
    
    LIMITATIONS (important for understanding):
    - Data is delayed 15-20 minutes for most symbols
    - Polling-based, not streaming
    - Rate limited (don't poll too frequently)
    - No true bid/ask, we estimate from last trade
    - No order book depth
    
    USE THIS FOR:
    - Research and strategy development
    - Understanding system architecture
    - Demonstrating concepts
    
    DO NOT USE FOR:
    - Live trading decisions
    - Latency-sensitive strategies
    - Market making
    """
    
    def __init__(self, config: MarketDataConfig, latency_config: LatencyConfig):
        super().__init__(config, latency_config)
        
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance not installed. Run: pip install yfinance"
            )
        
        self._poll_task: Optional[asyncio.Task] = None
        self._tickers: Dict[str, yf.Ticker] = {}
        
        # Cache for rate limiting
        self._last_fetch: Dict[str, datetime] = {}
    
    async def start(self) -> None:
        """Start polling Yahoo Finance for data."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize tickers
        for symbol in self._config.symbols:
            self._tickers[symbol] = yf.Ticker(symbol)
        
        # Start polling task
        self._poll_task = asyncio.create_task(self._poll_loop())
        
        logger.info(
            f"Started Yahoo Finance feed for {len(self._config.symbols)} symbols",
            category=LogCategory.MARKET_DATA
        )
    
    async def stop(self) -> None:
        """Stop the data feed."""
        self._running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped Yahoo Finance feed", category=LogCategory.MARKET_DATA)
    
    async def _poll_loop(self) -> None:
        """
        Main polling loop.
        
        In production, this would be replaced with:
        - Multicast UDP receiver
        - Kernel bypass for low latency
        - Hardware timestamping
        """
        while self._running:
            try:
                for symbol in self._config.symbols:
                    await self._fetch_quote(symbol)
                
                # Rate limiting - Yahoo Finance has limits
                await asyncio.sleep(self._config.poll_interval_seconds)
                
            except Exception as e:
                logger.error(
                    f"Error in poll loop: {e}",
                    category=LogCategory.MARKET_DATA
                )
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _fetch_quote(self, symbol: str) -> None:
        """
        Fetch latest quote for a symbol.
        
        Yahoo Finance doesn't provide true bid/ask for most symbols,
        so we estimate based on typical spread.
        """
        try:
            ticker = self._tickers[symbol]
            
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: ticker.info
            )
            
            # Extract price data
            # Yahoo Finance has different fields depending on market hours
            current_price = info.get('regularMarketPrice') or info.get('currentPrice') or 0
            
            if current_price <= 0:
                return
            
            # Estimate bid/ask from price and typical spread
            # Real HFT would have actual quotes
            estimated_spread_pct = 0.001  # 10 bps typical for liquid stocks
            half_spread = current_price * estimated_spread_pct / 2
            
            bid_price = current_price - half_spread
            ask_price = current_price + half_spread
            
            # Estimate size (this is entirely simulated)
            avg_volume = info.get('averageVolume', 1000000)
            typical_size = max(100, int(avg_volume / 10000))
            
            # Create quote
            now = datetime.utcnow()
            quote = Quote(
                symbol=symbol,
                bid_price=bid_price,
                bid_size=typical_size + random.randint(-50, 50),
                ask_price=ask_price,
                ask_size=typical_size + random.randint(-50, 50),
                timestamp=now,
                exchange_timestamp=now - timedelta(milliseconds=100),  # Simulated delay
                received_timestamp=now,
            )
            
            # Store latest
            with self._lock:
                self._latest_quotes[symbol] = quote
            
            # Create and publish event
            event = MarketDataEvent(
                event_type=MarketDataType.QUOTE,
                symbol=symbol,
                timestamp=now,
                data=quote,
            )
            
            await self._publish_event(event)
            
        except Exception as e:
            logger.warning(
                f"Failed to fetch quote for {symbol}: {e}",
                category=LogCategory.MARKET_DATA,
                symbol=symbol
            )


class SimulatedDataFeed(MarketDataFeed):
    """
    Simulated market data feed for backtesting.
    
    Generates realistic-looking market data with:
    - Configurable volatility and drift
    - Correlated movements between symbols
    - Microstructure effects (spread widening on volatility)
    
    This is essential for:
    - Deterministic backtesting
    - Stress testing (volatility shocks, gaps)
    - Testing edge cases
    """
    
    def __init__(
        self,
        config: MarketDataConfig,
        latency_config: LatencyConfig,
        seed: int = 42
    ):
        super().__init__(config, latency_config)
        
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._generate_task: Optional[asyncio.Task] = None
        
        # Price state for each symbol
        self._prices: Dict[str, float] = {}
        self._volatilities: Dict[str, float] = {}
        
        # Initialize with realistic starting prices
        self._initial_prices = {
            "AAPL": 185.0,
            "MSFT": 375.0,
            "GOOGL": 140.0,
            "SPY": 450.0,
            "AMZN": 150.0,
            "META": 350.0,
            "NVDA": 500.0,
            "TSLA": 250.0,
        }
        
        # Correlation matrix for realistic co-movement
        # SPY should be correlated with individual stocks
        self._correlation_matrix = None
    
    async def start(self) -> None:
        """Start generating simulated data."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize prices and volatilities
        for symbol in self._config.symbols:
            self._prices[symbol] = self._initial_prices.get(symbol, 100.0)
            self._volatilities[symbol] = 0.02  # 2% daily vol
        
        # Build correlation matrix
        self._build_correlation_matrix()
        
        # Start generation task
        self._generate_task = asyncio.create_task(self._generate_loop())
        
        logger.info(
            f"Started simulated feed for {len(self._config.symbols)} symbols",
            category=LogCategory.MARKET_DATA
        )
    
    async def stop(self) -> None:
        """Stop the simulated feed."""
        self._running = False
        
        if self._generate_task:
            self._generate_task.cancel()
            try:
                await self._generate_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped simulated feed", category=LogCategory.MARKET_DATA)
    
    def _build_correlation_matrix(self) -> None:
        """
        Build correlation matrix for multi-asset simulation.
        
        In reality, correlations are:
        - Time-varying (regime-dependent)
        - Asymmetric (higher in down markets)
        - Sector-dependent
        """
        n = len(self._config.symbols)
        
        # Start with identity matrix
        corr = np.eye(n)
        
        # Add correlations (simplified)
        for i in range(n):
            for j in range(i + 1, n):
                # Higher correlation between similar stocks
                base_corr = 0.5 + self._rng.uniform(-0.2, 0.2)
                corr[i, j] = base_corr
                corr[j, i] = base_corr
        
        # Ensure positive definiteness via Cholesky
        try:
            self._correlation_matrix = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Fallback to identity if matrix isn't PD
            self._correlation_matrix = np.eye(n)
    
    async def _generate_loop(self) -> None:
        """
        Generate market data ticks.
        
        This simulates the continuous arrival of market data
        as would happen with a real exchange feed.
        """
        tick_interval = 1.0 / self._config.simulated_ticks_per_second
        
        while self._running:
            try:
                # Generate correlated returns
                returns = self._generate_correlated_returns()
                
                # Update prices and generate quotes
                for i, symbol in enumerate(self._config.symbols):
                    self._update_price(symbol, returns[i])
                    await self._generate_quote(symbol)
                
                await asyncio.sleep(tick_interval)
                
            except Exception as e:
                logger.error(
                    f"Error in generate loop: {e}",
                    category=LogCategory.MARKET_DATA
                )
    
    def _generate_correlated_returns(self) -> np.ndarray:
        """
        Generate correlated returns using Cholesky decomposition.
        
        This is standard practice in multi-asset simulation:
        - Generate independent normals
        - Transform via Cholesky factor
        - Scale by volatility
        """
        n = len(self._config.symbols)
        
        # Independent standard normals
        z = self._rng.standard_normal(n)
        
        # Apply correlation structure
        if self._correlation_matrix is not None:
            z = self._correlation_matrix @ z
        
        # Scale by volatility (per-tick scaling)
        tick_vol = np.array([
            self._volatilities[s] / np.sqrt(252 * 6.5 * 3600 * self._config.simulated_ticks_per_second)
            for s in self._config.symbols
        ])
        
        return z * tick_vol
    
    def _update_price(self, symbol: str, return_: float) -> None:
        """
        Update price with geometric Brownian motion.
        
        GBM: dS = μSdt + σSdW
        
        We use log-normal returns to ensure prices stay positive.
        """
        self._prices[symbol] *= np.exp(return_)
    
    async def _generate_quote(self, symbol: str) -> None:
        """
        Generate a quote for a symbol.
        
        Includes microstructure effects:
        - Spread widens with volatility
        - Size varies with time of day
        - Occasional large size changes
        """
        price = self._prices[symbol]
        
        # Base spread in bps, widens with volatility
        base_spread_bps = 5.0  # 5 bps
        vol_multiplier = 1.0 + self._volatilities[symbol] * 10
        spread_bps = base_spread_bps * vol_multiplier
        
        # Convert to price
        spread = price * spread_bps / 10000
        half_spread = spread / 2
        
        # Slight asymmetry based on recent direction
        asymmetry = self._rng.uniform(-0.1, 0.1) * half_spread
        
        bid_price = price - half_spread + asymmetry
        ask_price = price + half_spread + asymmetry
        
        # Size simulation
        base_size = 500
        size_noise = int(self._rng.exponential(200))
        bid_size = base_size + size_noise + int(self._rng.normal(0, 50))
        ask_size = base_size + size_noise + int(self._rng.normal(0, 50))
        
        # Ensure positive
        bid_size = max(100, bid_size)
        ask_size = max(100, ask_size)
        
        # Create quote
        now = datetime.utcnow()
        quote = Quote(
            symbol=symbol,
            bid_price=round(bid_price, 4),
            bid_size=bid_size,
            ask_price=round(ask_price, 4),
            ask_size=ask_size,
            timestamp=now,
            exchange_timestamp=now,
            received_timestamp=now,
        )
        
        # Store latest
        with self._lock:
            self._latest_quotes[symbol] = quote
        
        # Create and publish event
        event = MarketDataEvent(
            event_type=MarketDataType.QUOTE,
            symbol=symbol,
            timestamp=now,
            data=quote,
        )
        
        await self._publish_event(event)
    
    def inject_volatility_shock(self, multiplier: float = 3.0) -> None:
        """
        Inject a volatility shock for stress testing.
        
        This simulates events like:
        - Flash crashes
        - News announcements
        - Market opens
        """
        for symbol in self._volatilities:
            self._volatilities[symbol] *= multiplier
        
        logger.warning(
            f"Injected volatility shock: {multiplier}x",
            category=LogCategory.MARKET_DATA
        )
    
    def inject_price_gap(self, symbol: str, gap_pct: float) -> None:
        """
        Inject a price gap for stress testing.
        
        This simulates:
        - Overnight gaps
        - Circuit breaker halts
        - Large block trades
        """
        if symbol in self._prices:
            old_price = self._prices[symbol]
            self._prices[symbol] *= (1 + gap_pct)
            logger.warning(
                f"Injected price gap for {symbol}: {old_price:.2f} -> {self._prices[symbol]:.2f}",
                category=LogCategory.MARKET_DATA,
                symbol=symbol
            )


def create_data_feed(
    config: MarketDataConfig,
    latency_config: LatencyConfig
) -> MarketDataFeed:
    """
    Factory function to create appropriate data feed.
    
    This pattern allows easy swapping of data sources
    without changing consumer code.
    """
    if config.source == DataSource.YAHOO_FINANCE:
        return YahooFinanceDataFeed(config, latency_config)
    elif config.source == DataSource.SIMULATED:
        return SimulatedDataFeed(config, latency_config)
    else:
        raise ValueError(f"Unknown data source: {config.source}")
