"""
Latency Arbitrage Signal for HFT Research System
=================================================

Latency arbitrage exploits the time delay between when
information hits different venues or instruments.

WHAT IS LATENCY ARBITRAGE?
==========================

When news or a large order moves the price of one asset,
related assets should move too. But this takes TIME:
- Information propagates through networks
- Different venues update at different speeds
- Algorithms need time to compute and react

A faster trader can:
1. Observe price change in Asset A
2. Predict that Asset B will move (correlation)
3. Trade Asset B before others react
4. Profit from the predictable price movement

EXAMPLES:
1. SPY vs Futures: E-mini S&P futures often lead SPY ETF
2. Cross-listed stocks: NYSE vs LSE prices can diverge briefly
3. ETF vs Basket: ETF price vs NAV of underlying stocks
4. ADRs vs Home listing: Arbitrage across time zones

ETHICAL CONSIDERATIONS:
=======================
Latency arbitrage is controversial:
- Proponents: Provides liquidity, improves price efficiency
- Critics: "Front-running" other traders, increases costs

This implementation is for EDUCATIONAL purposes to understand
the mechanics and detection of such strategies.

IMPLEMENTATION:
==============
We simulate latency arbitrage detection between:
- An "index" (e.g., SPY) and its "components"
- Apply artificial latency offsets
- Detect when prices are temporarily stale
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque, Tuple, Set

from ..data.market_data_feed import Quote
from ..infra.logging import get_logger, LogCategory
from ..infra.config import SignalConfig


logger = get_logger()


@dataclass
class LatencyArbSignal:
    """
    Output of latency arbitrage detection.
    
    Contains information about detected price staleness
    and potential arbitrage opportunities.
    """
    symbol: str                  # The "stale" symbol
    reference_symbol: str        # The "fast" reference
    timestamp: datetime
    
    # Core signal
    signal_value: float          # -1 to +1, direction to trade
    staleness_score: float       # How stale is the price (0-1)
    predicted_move_bps: float    # Expected move in basis points
    
    # Prices
    current_price: float         # Current price of stale symbol
    fair_value: float            # Estimated fair value based on reference
    deviation_bps: float         # Current deviation from fair value
    
    # Confidence
    confidence: float
    correlation: float           # Historical correlation
    latency_ms: float            # Detected latency
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "reference_symbol": self.reference_symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_value": self.signal_value,
            "staleness_score": self.staleness_score,
            "predicted_move_bps": self.predicted_move_bps,
            "deviation_bps": self.deviation_bps,
            "confidence": self.confidence,
        }


@dataclass
class SymbolPair:
    """
    Configuration for a pair of related symbols.
    """
    fast_symbol: str          # Symbol that moves first (e.g., futures)
    slow_symbol: str          # Symbol that lags (e.g., ETF)
    correlation: float        # Historical correlation
    beta: float               # Beta coefficient (how much slow moves per unit of fast)
    typical_lag_ms: float     # Typical lag in milliseconds
    
    # Thresholds
    min_deviation_bps: float = 5.0    # Minimum deviation to trigger
    max_deviation_bps: float = 50.0   # Maximum (avoid false signals on data errors)


class LatencyArbitrageDetector:
    """
    Detects latency arbitrage opportunities.
    
    This class monitors related instruments and identifies when:
    1. The "fast" instrument has moved
    2. The "slow" instrument hasn't caught up yet
    3. There's a predictable arbitrage opportunity
    
    SIMULATED NATURE:
    =================
    With free data (Yahoo Finance), true latency arbitrage is impossible.
    This implementation:
    - Simulates artificial latency between symbols
    - Demonstrates the detection methodology
    - Provides a framework for real implementation
    
    IN PRODUCTION:
    - Would use microsecond timestamps
    - Co-located feeds for each venue
    - Hardware timestamping
    - Sub-microsecond execution
    """
    
    def __init__(self, config: SignalConfig):
        """
        Initialize latency arbitrage detector.
        
        Args:
            config: Signal configuration
        """
        self._config = config
        
        # Symbol pairs to monitor
        self._pairs: Dict[str, SymbolPair] = {}
        
        # Price history for each symbol
        self._price_history: Dict[str, Deque[Tuple[datetime, float]]] = {}
        
        # Correlation tracking
        self._return_history: Dict[str, Deque[float]] = {}
        self._correlations: Dict[Tuple[str, str], float] = {}
        
        # Latest prices and timestamps
        self._latest_prices: Dict[str, float] = {}
        self._latest_timestamps: Dict[str, datetime] = {}
        
        # Simulated latency offsets (for demonstration)
        self._latency_offsets: Dict[str, float] = {}
        
        # Parameters
        self._correlation_threshold = config.correlation_threshold
        self._staleness_threshold_ms = config.staleness_threshold_ms
    
    def add_pair(
        self,
        fast_symbol: str,
        slow_symbol: str,
        correlation: float = 0.9,
        beta: float = 1.0,
        typical_lag_ms: float = 10.0
    ) -> None:
        """
        Add a symbol pair to monitor for latency arbitrage.
        
        Args:
            fast_symbol: The faster-updating symbol (e.g., futures)
            slow_symbol: The slower-updating symbol (e.g., ETF)
            correlation: Expected correlation between returns
            beta: How much slow moves per unit of fast
            typical_lag_ms: Expected lag in milliseconds
        """
        pair = SymbolPair(
            fast_symbol=fast_symbol,
            slow_symbol=slow_symbol,
            correlation=correlation,
            beta=beta,
            typical_lag_ms=typical_lag_ms,
        )
        
        self._pairs[slow_symbol] = pair
        
        # Initialize history
        for symbol in [fast_symbol, slow_symbol]:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=1000)
                self._return_history[symbol] = deque(maxlen=1000)
                self._latency_offsets[symbol] = 0.0
        
        # Set simulated latency for slow symbol
        self._latency_offsets[slow_symbol] = typical_lag_ms
        
        logger.info(
            f"Added latency arb pair: {fast_symbol} -> {slow_symbol} (lag={typical_lag_ms}ms)",
            category=LogCategory.SIGNAL
        )
    
    def add_default_pairs(self) -> None:
        """
        Add default pairs for common latency arbitrage scenarios.
        
        These are simplified examples - real correlations and betas
        would be estimated from historical data.
        """
        # SPY leads/lags with its components
        # In reality, futures (ES) lead SPY, and SPY leads components
        self.add_pair("SPY", "AAPL", correlation=0.85, beta=1.2, typical_lag_ms=20.0)
        self.add_pair("SPY", "MSFT", correlation=0.80, beta=1.0, typical_lag_ms=20.0)
        self.add_pair("SPY", "GOOGL", correlation=0.75, beta=1.1, typical_lag_ms=25.0)
    
    def update_price(self, quote: Quote) -> Optional[LatencyArbSignal]:
        """
        Update with new price and check for arbitrage opportunity.
        
        This is the main entry point for price updates.
        """
        symbol = quote.symbol
        price = quote.mid_price
        timestamp = quote.timestamp
        
        # Apply simulated latency offset
        latency_ms = self._latency_offsets.get(symbol, 0.0)
        
        # Store in history
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)
            self._return_history[symbol] = deque(maxlen=1000)
        
        # Calculate return
        if self._latest_prices.get(symbol):
            ret = (price / self._latest_prices[symbol]) - 1
            self._return_history[symbol].append(ret)
        
        self._price_history[symbol].append((timestamp, price))
        self._latest_prices[symbol] = price
        self._latest_timestamps[symbol] = timestamp
        
        # Check if this is a slow symbol we're monitoring
        if symbol in self._pairs:
            return self._check_arbitrage(symbol, timestamp, latency_ms)
        
        # Also update correlations if this is a fast symbol
        for pair in self._pairs.values():
            if symbol == pair.fast_symbol:
                self._update_correlation(pair)
        
        return None
    
    def _check_arbitrage(
        self,
        slow_symbol: str,
        timestamp: datetime,
        latency_ms: float
    ) -> Optional[LatencyArbSignal]:
        """
        Check for latency arbitrage opportunity.
        
        The algorithm:
        1. Get recent price change in fast symbol
        2. Estimate what slow symbol "should" be priced at
        3. Compare to actual price
        4. If deviation > threshold, signal opportunity
        """
        pair = self._pairs[slow_symbol]
        fast_symbol = pair.fast_symbol
        
        # Need prices for both symbols
        if fast_symbol not in self._latest_prices or slow_symbol not in self._latest_prices:
            return None
        
        fast_price = self._latest_prices[fast_symbol]
        slow_price = self._latest_prices[slow_symbol]
        
        # Calculate recent return in fast symbol
        fast_history = self._price_history[fast_symbol]
        if len(fast_history) < 2:
            return None
        
        # Get price from ~latency_ms ago
        lookback_time = timestamp - timedelta(milliseconds=latency_ms * 2)
        
        past_fast_price = None
        for ts, price in reversed(fast_history):
            if ts <= lookback_time:
                past_fast_price = price
                break
        
        if past_fast_price is None or past_fast_price == 0:
            return None
        
        # Fast symbol return since the "stale" time
        fast_return = (fast_price / past_fast_price) - 1
        
        # Predicted return for slow symbol
        predicted_slow_return = fast_return * pair.beta
        
        # Get slow symbol's price from lookback time
        slow_history = self._price_history[slow_symbol]
        past_slow_price = None
        for ts, price in reversed(slow_history):
            if ts <= lookback_time:
                past_slow_price = price
                break
        
        if past_slow_price is None:
            past_slow_price = slow_price  # Use current as fallback
        
        # Estimate fair value
        fair_value = past_slow_price * (1 + predicted_slow_return)
        
        # Calculate deviation
        if fair_value > 0:
            deviation_bps = (slow_price - fair_value) / fair_value * 10000
        else:
            deviation_bps = 0.0
        
        # Check if deviation is significant
        if abs(deviation_bps) < pair.min_deviation_bps:
            return None
        
        # Avoid obviously erroneous signals
        if abs(deviation_bps) > pair.max_deviation_bps:
            return None
        
        # Calculate staleness score (how confident we are price is stale)
        staleness_score = self._calculate_staleness(
            slow_symbol, fast_symbol, latency_ms
        )
        
        # Predicted move: price should converge to fair value
        predicted_move_bps = -deviation_bps  # Negative because if slow is too high, it should fall
        
        # Signal direction
        if deviation_bps > 0:
            # Slow symbol is too high, should fall -> SELL
            signal_value = -min(abs(deviation_bps) / 20, 1.0)
        else:
            # Slow symbol is too low, should rise -> BUY
            signal_value = min(abs(deviation_bps) / 20, 1.0)
        
        # Confidence based on correlation and staleness
        correlation = self._correlations.get((fast_symbol, slow_symbol), pair.correlation)
        confidence = self._calculate_confidence(
            staleness_score, correlation, deviation_bps, pair
        )
        
        return LatencyArbSignal(
            symbol=slow_symbol,
            reference_symbol=fast_symbol,
            timestamp=timestamp,
            signal_value=signal_value,
            staleness_score=staleness_score,
            predicted_move_bps=predicted_move_bps,
            current_price=slow_price,
            fair_value=fair_value,
            deviation_bps=deviation_bps,
            confidence=confidence,
            correlation=correlation,
            latency_ms=latency_ms,
        )
    
    def _calculate_staleness(
        self,
        slow_symbol: str,
        fast_symbol: str,
        latency_ms: float
    ) -> float:
        """
        Calculate how "stale" the slow price appears.
        
        Staleness is higher when:
        1. Fast symbol has moved significantly recently
        2. Slow symbol hasn't moved
        3. Update timestamps are divergent
        """
        # Compare update rates
        slow_history = self._price_history[slow_symbol]
        fast_history = self._price_history[fast_symbol]
        
        if len(slow_history) < 2 or len(fast_history) < 2:
            return 0.5
        
        # Recent price changes
        slow_prices = [p for _, p in list(slow_history)[-10:]]
        fast_prices = [p for _, p in list(fast_history)[-10:]]
        
        if len(slow_prices) < 2 or len(fast_prices) < 2:
            return 0.5
        
        slow_std = np.std(slow_prices) / (np.mean(slow_prices) + 0.0001)
        fast_std = np.std(fast_prices) / (np.mean(fast_prices) + 0.0001)
        
        # If fast is moving more than slow, slow might be stale
        if fast_std > 0:
            staleness = min(fast_std / max(slow_std, 0.0001), 2.0) / 2.0
        else:
            staleness = 0.5
        
        # Adjust by latency
        latency_factor = min(latency_ms / self._staleness_threshold_ms, 1.0)
        staleness = staleness * 0.5 + latency_factor * 0.5
        
        return np.clip(staleness, 0.0, 1.0)
    
    def _calculate_confidence(
        self,
        staleness_score: float,
        correlation: float,
        deviation_bps: float,
        pair: SymbolPair
    ) -> float:
        """
        Calculate confidence in the latency arb signal.
        """
        confidence = 0.3  # Base confidence (latency arb is risky)
        
        # Higher correlation = higher confidence
        if correlation > self._correlation_threshold:
            confidence += (correlation - 0.5) * 0.3
        
        # Higher staleness = higher confidence
        confidence += staleness_score * 0.2
        
        # Deviation in sweet spot
        if pair.min_deviation_bps < abs(deviation_bps) < 20:
            confidence += 0.15
        elif abs(deviation_bps) > 30:
            confidence -= 0.1  # Too large might be data error
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _update_correlation(self, pair: SymbolPair) -> None:
        """
        Update correlation estimate between pair symbols.
        """
        fast_returns = list(self._return_history.get(pair.fast_symbol, []))
        slow_returns = list(self._return_history.get(pair.slow_symbol, []))
        
        # Need sufficient data
        min_len = min(len(fast_returns), len(slow_returns))
        if min_len < 30:
            return
        
        # Align and calculate correlation
        fast_arr = np.array(fast_returns[-min_len:])
        slow_arr = np.array(slow_returns[-min_len:])
        
        # Handle edge cases
        if np.std(fast_arr) < 0.0001 or np.std(slow_arr) < 0.0001:
            return
        
        correlation = np.corrcoef(fast_arr, slow_arr)[0, 1]
        
        if not np.isnan(correlation):
            self._correlations[(pair.fast_symbol, pair.slow_symbol)] = correlation
    
    def set_latency_offset(self, symbol: str, latency_ms: float) -> None:
        """
        Set simulated latency offset for a symbol.
        
        Used for stress testing different latency scenarios.
        """
        self._latency_offsets[symbol] = latency_ms
        logger.debug(
            f"Set latency offset for {symbol}: {latency_ms}ms",
            category=LogCategory.SIGNAL,
            symbol=symbol
        )
    
    def get_correlations(self) -> Dict[Tuple[str, str], float]:
        """Get current correlation estimates."""
        return dict(self._correlations)
    
    def reset(self) -> None:
        """Reset all state."""
        self._price_history.clear()
        self._return_history.clear()
        self._correlations.clear()
        self._latest_prices.clear()
        self._latest_timestamps.clear()
