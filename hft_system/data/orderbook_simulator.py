"""
Order Book Simulator for HFT Research System
=============================================

This module provides a realistic order book simulation:
- Multiple price levels with quantities
- Order arrival/cancellation dynamics
- Queue position modeling
- Market impact estimation

ORDER BOOK FUNDAMENTALS:
========================

An order book is the core data structure in electronic markets:

    BIDS (Buy Orders)              ASKS (Sell Orders)
    Price    |  Size               Price    |  Size
    ─────────┼───────              ─────────┼───────
    100.02   |  500   <-- Best     100.03   |  300   <-- Best
    100.01   |  1200               100.04   |  800
    100.00   |  2500               100.05   |  1500
    99.99    |  3000               100.06   |  2000

Key concepts:
- Spread: Best ask - Best bid (100.03 - 100.02 = $0.01)
- Mid price: (Best bid + Best ask) / 2 = 100.025
- Book depth: Sum of sizes at all levels
- Imbalance: Relative size of bids vs asks

WHY ORDER BOOK MATTERS FOR HFT:
1. Microprice signals: Weighted average of bid/ask by size
2. Queue position: Where you are in line affects fill probability
3. Market impact: How your order moves the price
4. Adverse selection: Risk of trading against informed flow

WHAT THIS SIMULATOR DOES:
- Maintains multi-level order book state
- Simulates realistic order flow dynamics
- Estimates queue position for limit orders
- Computes market impact for market orders

WHAT IT DOESN'T DO (would need real data):
- True L3 data (individual orders)
- Actual queue position tracking
- Real hidden order detection
- Accurate cross-venue arbitrage
"""

import asyncio
import heapq
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
import threading

import numpy as np

from ..infra.logging import get_logger, LogCategory


logger = get_logger()


class OrderSide(Enum):
    """Order side."""
    BID = "BID"
    ASK = "ASK"


@dataclass
class PriceLevel:
    """
    Represents a single price level in the order book.
    
    In real L3 data, each level would contain:
    - List of individual orders with timestamps
    - Queue position for each order
    - Hidden order estimates
    
    We simplify to aggregate size at each level.
    """
    price: float
    size: int
    num_orders: int = 1  # Estimated number of orders at this level
    
    # Queue dynamics
    arrival_rate: float = 10.0   # Orders arriving per second
    cancel_rate: float = 5.0    # Orders canceling per second
    
    def __lt__(self, other: 'PriceLevel') -> bool:
        """For heap ordering."""
        return self.price < other.price
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PriceLevel):
            return NotImplemented
        return self.price == other.price


@dataclass
class OrderBook:
    """
    Full order book state for a single symbol.
    
    This is a simplified representation. Real order books:
    - Update millions of times per day
    - Require nanosecond precision
    - Have complex message types (add, modify, delete, execute)
    """
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Sorted price levels
    bids: List[PriceLevel] = field(default_factory=list)  # Highest to lowest
    asks: List[PriceLevel] = field(default_factory=list)  # Lowest to highest
    
    # Sequence for change detection
    sequence_number: int = 0
    
    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Best (highest) bid price level."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Best (lowest) ask price level."""
        return self.asks[0] if self.asks else None
    
    @property
    def best_bid_price(self) -> float:
        """Best bid price."""
        return self.best_bid.price if self.best_bid else 0.0
    
    @property
    def best_ask_price(self) -> float:
        """Best ask price."""
        return self.best_ask.price if self.best_ask else float('inf')
    
    @property
    def mid_price(self) -> float:
        """Mid-point between best bid and ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid_price + self.best_ask_price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread in dollars."""
        if self.best_bid and self.best_ask:
            return self.best_ask_price - self.best_bid_price
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price > 0:
            return self.spread / self.mid_price * 10000
        return 0.0
    
    @property
    def microprice(self) -> float:
        """
        Microprice: Size-weighted mid price.
        
        This is a key HFT signal! The microprice adjusts the mid-price
        based on the relative sizes at the best bid and ask.
        
        If there's more size on the bid side, microprice is above mid
        (more buying pressure suggests price will go up).
        
        Formula: microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        
        This is essentially a predictor of the next trade direction.
        """
        if not self.best_bid or not self.best_ask:
            return self.mid_price
        
        bid_size = self.best_bid.size
        ask_size = self.best_ask.size
        
        if bid_size + ask_size == 0:
            return self.mid_price
        
        return (bid_size * self.best_ask_price + ask_size * self.best_bid_price) / (bid_size + ask_size)
    
    @property
    def book_imbalance(self) -> float:
        """
        Order book imbalance across all levels.
        
        Positive = more bids than asks (bullish)
        Negative = more asks than bids (bearish)
        Range: -1 to +1
        """
        total_bid_size = sum(level.size for level in self.bids)
        total_ask_size = sum(level.size for level in self.asks)
        total = total_bid_size + total_ask_size
        
        if total == 0:
            return 0.0
        
        return (total_bid_size - total_ask_size) / total
    
    @property
    def top_of_book_imbalance(self) -> float:
        """
        Imbalance at best bid/ask only.
        
        This is a faster-moving signal than full book imbalance.
        """
        if not self.best_bid or not self.best_ask:
            return 0.0
        
        bid_size = self.best_bid.size
        ask_size = self.best_ask.size
        total = bid_size + ask_size
        
        if total == 0:
            return 0.0
        
        return (bid_size - ask_size) / total
    
    def depth_at_levels(self, n_levels: int = 5) -> Tuple[int, int]:
        """
        Total depth at top N levels on each side.
        
        Used for understanding liquidity available.
        """
        bid_depth = sum(level.size for level in self.bids[:n_levels])
        ask_depth = sum(level.size for level in self.asks[:n_levels])
        return bid_depth, ask_depth
    
    def price_for_size(self, side: OrderSide, size: int) -> float:
        """
        Calculate the average price to execute a given size.
        
        This is the market impact calculation - how much does
        our order move the market?
        
        A market buy of 1000 shares might need to "walk up the book":
        - 300 @ 100.03 (best ask)
        - 500 @ 100.04
        - 200 @ 100.05
        - Average price: (300*100.03 + 500*100.04 + 200*100.05) / 1000
        """
        levels = self.asks if side == OrderSide.BID else self.bids
        
        remaining = size
        total_cost = 0.0
        
        for level in levels:
            executed = min(remaining, level.size)
            total_cost += executed * level.price
            remaining -= executed
            
            if remaining <= 0:
                break
        
        if size - remaining > 0:
            return total_cost / (size - remaining)
        return 0.0
    
    def market_impact_bps(self, side: OrderSide, size: int) -> float:
        """
        Estimate market impact in basis points.
        
        Impact = (execution price - mid price) / mid price * 10000
        """
        if self.mid_price <= 0:
            return 0.0
        
        exec_price = self.price_for_size(side, size)
        
        if side == OrderSide.BID:  # Buying, price goes up
            return (exec_price - self.mid_price) / self.mid_price * 10000
        else:  # Selling, price goes down
            return (self.mid_price - exec_price) / self.mid_price * 10000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "best_bid": self.best_bid_price,
            "best_ask": self.best_ask_price,
            "mid_price": self.mid_price,
            "microprice": self.microprice,
            "spread_bps": self.spread_bps,
            "imbalance": self.book_imbalance,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
        }


class OrderBookSimulator:
    """
    Simulates order book dynamics for backtesting.
    
    This class generates realistic order book evolution:
    - Price levels change based on quotes
    - Sizes evolve with Poisson arrival/cancellation
    - Spread dynamics respond to volatility
    
    LIMITATIONS:
    - Not using real order flow data
    - Simplified queue dynamics
    - No actual order ID tracking
    
    HOW TO IMPROVE FOR PRODUCTION:
    1. Use LOBSTER or similar historical L3 data
    2. Implement actual order matching engine
    3. Track individual order positions
    4. Model hidden orders and icebergs
    """
    
    def __init__(
        self,
        symbols: List[str],
        num_levels: int = 10,
        tick_size: float = 0.01,
        seed: int = 42
    ):
        """
        Initialize order book simulator.
        
        Args:
            symbols: List of symbols to simulate
            num_levels: Number of price levels on each side
            tick_size: Minimum price increment
            seed: Random seed for reproducibility
        """
        self._symbols = symbols
        self._num_levels = num_levels
        self._tick_size = tick_size
        self._rng = np.random.default_rng(seed)
        
        # Order books for each symbol
        self._books: Dict[str, OrderBook] = {}
        
        # Parameters for dynamics
        self._volatilities: Dict[str, float] = {}
        self._base_sizes: Dict[str, int] = {}
        
        # Queue position tracking (simplified)
        self._queue_positions: Dict[str, Dict[float, Deque[Tuple[str, int]]]] = {}
        
        self._lock = threading.Lock()
        self._sequence = 0
    
    def initialize_book(
        self,
        symbol: str,
        mid_price: float,
        spread_bps: float = 5.0,
        volatility: float = 0.02,
        base_size: int = 500
    ) -> OrderBook:
        """
        Initialize order book for a symbol.
        
        Creates a realistic starting state with:
        - Multiple levels at tick increments
        - Sizes that increase away from mid (typical pattern)
        - Some randomness in sizes
        """
        self._volatilities[symbol] = volatility
        self._base_sizes[symbol] = base_size
        
        # Calculate spread
        spread = mid_price * spread_bps / 10000
        half_spread = spread / 2
        
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Round to tick size
        best_bid = round(best_bid / self._tick_size) * self._tick_size
        best_ask = round(best_ask / self._tick_size) * self._tick_size
        
        # Ensure minimum spread
        if best_ask <= best_bid:
            best_ask = best_bid + self._tick_size
        
        # Generate bid levels (decreasing prices)
        bids = []
        for i in range(self._num_levels):
            price = best_bid - i * self._tick_size
            # Size increases away from mid (typical pattern)
            size_multiplier = 1.0 + i * 0.3
            size = int(base_size * size_multiplier * (1 + self._rng.uniform(-0.2, 0.2)))
            bids.append(PriceLevel(price=round(price, 4), size=max(100, size)))
        
        # Generate ask levels (increasing prices)
        asks = []
        for i in range(self._num_levels):
            price = best_ask + i * self._tick_size
            size_multiplier = 1.0 + i * 0.3
            size = int(base_size * size_multiplier * (1 + self._rng.uniform(-0.2, 0.2)))
            asks.append(PriceLevel(price=round(price, 4), size=max(100, size)))
        
        book = OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            sequence_number=self._get_next_sequence()
        )
        
        with self._lock:
            self._books[symbol] = book
        
        logger.debug(
            f"Initialized order book for {symbol}: mid={book.mid_price:.4f}, spread={book.spread_bps:.1f}bps",
            category=LogCategory.MARKET_DATA,
            symbol=symbol
        )
        
        return book
    
    def update_from_quote(
        self,
        symbol: str,
        bid_price: float,
        bid_size: int,
        ask_price: float,
        ask_size: int
    ) -> OrderBook:
        """
        Update order book from a new quote.
        
        This adjusts the book around the new best bid/ask
        while maintaining realistic deeper levels.
        """
        with self._lock:
            book = self._books.get(symbol)
            if not book:
                return self.initialize_book(
                    symbol,
                    mid_price=(bid_price + ask_price) / 2
                )
            
            # Update best levels
            if book.bids:
                book.bids[0] = PriceLevel(price=bid_price, size=bid_size)
            else:
                book.bids.append(PriceLevel(price=bid_price, size=bid_size))
            
            if book.asks:
                book.asks[0] = PriceLevel(price=ask_price, size=ask_size)
            else:
                book.asks.append(PriceLevel(price=ask_price, size=ask_size))
            
            # Rebuild deeper levels around new mid
            base_size = self._base_sizes.get(symbol, 500)
            
            # Regenerate bid levels
            new_bids = [book.bids[0]]
            for i in range(1, self._num_levels):
                price = bid_price - i * self._tick_size
                # Find existing level or create new
                existing = next((l for l in book.bids if abs(l.price - price) < 0.001), None)
                if existing:
                    # Evolve size slightly
                    new_size = existing.size + int(self._rng.normal(0, 50))
                    new_bids.append(PriceLevel(price=round(price, 4), size=max(100, new_size)))
                else:
                    size_multiplier = 1.0 + i * 0.3
                    size = int(base_size * size_multiplier * (1 + self._rng.uniform(-0.2, 0.2)))
                    new_bids.append(PriceLevel(price=round(price, 4), size=max(100, size)))
            
            # Regenerate ask levels
            new_asks = [book.asks[0]]
            for i in range(1, self._num_levels):
                price = ask_price + i * self._tick_size
                existing = next((l for l in book.asks if abs(l.price - price) < 0.001), None)
                if existing:
                    new_size = existing.size + int(self._rng.normal(0, 50))
                    new_asks.append(PriceLevel(price=round(price, 4), size=max(100, new_size)))
                else:
                    size_multiplier = 1.0 + i * 0.3
                    size = int(base_size * size_multiplier * (1 + self._rng.uniform(-0.2, 0.2)))
                    new_asks.append(PriceLevel(price=round(price, 4), size=max(100, size)))
            
            book.bids = new_bids
            book.asks = new_asks
            book.timestamp = datetime.utcnow()
            book.sequence_number = self._get_next_sequence()
            
            return book
    
    def simulate_trade_impact(
        self,
        symbol: str,
        side: OrderSide,
        size: int
    ) -> Tuple[float, OrderBook]:
        """
        Simulate the impact of a trade on the order book.
        
        When a market order executes:
        1. It consumes liquidity at the best level
        2. If size > best level, it walks through multiple levels
        3. The best level changes, widening the spread temporarily
        
        Returns:
            Tuple of (average fill price, updated order book)
        """
        with self._lock:
            book = self._books.get(symbol)
            if not book:
                raise ValueError(f"No order book for {symbol}")
            
            # Which side gets consumed
            levels = book.asks if side == OrderSide.BID else book.bids
            
            remaining = size
            total_cost = 0.0
            executed = 0
            
            new_levels = []
            for level in levels:
                if remaining <= 0:
                    new_levels.append(level)
                    continue
                
                if level.size <= remaining:
                    # Consume entire level
                    total_cost += level.size * level.price
                    executed += level.size
                    remaining -= level.size
                    # Level disappears
                else:
                    # Partial fill at this level
                    total_cost += remaining * level.price
                    executed += remaining
                    new_levels.append(PriceLevel(
                        price=level.price,
                        size=level.size - remaining
                    ))
                    remaining = 0
            
            # Update the book
            if side == OrderSide.BID:
                book.asks = new_levels if new_levels else [levels[-1]]
            else:
                book.bids = new_levels if new_levels else [levels[-1]]
            
            book.sequence_number = self._get_next_sequence()
            
            avg_price = total_cost / executed if executed > 0 else 0.0
            
            return avg_price, book
    
    def get_book(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for a symbol."""
        with self._lock:
            return self._books.get(symbol)
    
    def estimate_queue_position(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        order_size: int
    ) -> float:
        """
        Estimate queue position for a limit order.
        
        Queue position determines fill probability:
        - Position 0: First in line, fills first
        - Higher positions: Wait longer, may not fill
        
        In real markets, queue position depends on:
        - Time priority (first come, first served)
        - Price-time priority (better price = front of queue)
        - Hidden order rules
        
        We estimate based on size at the price level.
        
        Returns:
            Estimated fraction of queue ahead of us (0 to 1)
        """
        with self._lock:
            book = self._books.get(symbol)
            if not book:
                return 0.5
            
            levels = book.bids if side == OrderSide.BID else book.asks
            
            # Find the level at this price
            for level in levels:
                if abs(level.price - price) < 0.001:
                    # Assume we're in the middle on average
                    # Real system would track actual queue position
                    return 0.5
            
            # Price not in book - we'd be at front of new level
            return 0.0
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence += 1
        return self._sequence


class MarketImpactModel:
    """
    Models market impact for order execution.
    
    Market impact is the price movement caused by trading:
    - Temporary impact: Immediate price move, reverts quickly
    - Permanent impact: Lasting effect from information content
    
    SQUARE ROOT MODEL (Almgren-Chriss):
    impact = σ * (Q / V)^0.5 * sign(Q)
    
    where:
    - σ = volatility
    - Q = order quantity
    - V = average daily volume
    
    This is a simplified version. Real models consider:
    - Time of day effects
    - Momentum/mean reversion
    - Order type (limit vs market)
    - Venue characteristics
    """
    
    def __init__(
        self,
        temporary_impact_coefficient: float = 0.1,
        permanent_impact_coefficient: float = 0.05,
        volatility_multiplier: float = 1.0
    ):
        self._temp_coef = temporary_impact_coefficient
        self._perm_coef = permanent_impact_coefficient
        self._vol_mult = volatility_multiplier
    
    def estimate_impact(
        self,
        side: OrderSide,
        size: int,
        volatility: float,
        avg_daily_volume: int,
        spread_bps: float
    ) -> Dict[str, float]:
        """
        Estimate market impact for an order.
        
        Args:
            side: Buy or sell
            size: Order size in shares
            volatility: Daily volatility (e.g., 0.02 for 2%)
            avg_daily_volume: Average daily volume
            spread_bps: Current spread in basis points
        
        Returns:
            Dictionary with impact estimates in basis points
        """
        if avg_daily_volume <= 0:
            avg_daily_volume = 1000000  # Default
        
        # Participation rate
        participation = size / avg_daily_volume
        
        # Square root model for impact
        sqrt_participation = np.sqrt(participation)
        
        # Temporary impact (reverts)
        temp_impact = self._temp_coef * volatility * sqrt_participation * 10000 * self._vol_mult
        
        # Permanent impact (persists)
        perm_impact = self._perm_coef * volatility * sqrt_participation * 10000 * self._vol_mult
        
        # Total impact
        total_impact = temp_impact + perm_impact
        
        # Add spread crossing cost if market order
        half_spread = spread_bps / 2
        
        # Sign convention: positive impact = cost (adverse to trader)
        direction = 1 if side == OrderSide.BID else -1
        
        return {
            "temporary_impact_bps": temp_impact,
            "permanent_impact_bps": perm_impact,
            "total_impact_bps": total_impact,
            "spread_cost_bps": half_spread,
            "all_in_cost_bps": total_impact + half_spread,
            "participation_rate": participation,
        }
