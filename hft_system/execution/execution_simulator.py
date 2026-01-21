"""
Execution Simulator for HFT Research System
============================================

This module simulates order execution with realistic effects:
- Slippage (market impact)
- Partial fills
- Queue position and fill probability
- Adverse selection
- Latency

EXECUTION SIMULATION IS CRITICAL:
=================================

Backtests are only as good as their execution simulation.
Naive "fill at mid" assumptions lead to:
- Overestimated profits
- Underestimated risk
- Strategies that fail in production

WHAT WE SIMULATE:
1. SLIPPAGE:
   - Immediate impact from crossing spread
   - Market impact from large orders
   - Volatility-adjusted slippage

2. PARTIAL FILLS:
   - Limit orders may not fill completely
   - Based on queue position and liquidity
   - Time-based fill probability decay

3. ADVERSE SELECTION:
   - Limit orders get picked off by informed traders
   - Fills are more likely when price moves against you
   - Winners are harder to fill, losers fill easily

4. LATENCY:
   - Time from order to fill
   - Stale prices during execution
   - Race conditions with other orders

PRODUCTION DIFFERENCES:
- Real exchanges have complex matching rules
- Hidden orders and icebergs
- Multiple venues with different characteristics
- Regulatory constraints (trade-through rules)
"""

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Awaitable

import numpy as np

from .smart_order_router import (
    Order, OrderType, OrderStatus, OrderSideEnum, SmartOrderRouter
)
from ..data.orderbook_simulator import OrderBook, OrderSide, MarketImpactModel
from ..data.market_data_feed import Quote
from ..infra.config import ExecutionConfig, LatencyConfig
from ..infra.logging import get_logger, LogCategory
from ..infra.performance_metrics import Trade, Side, PerformanceTracker


logger = get_logger()


@dataclass
class ExecutionResult:
    """
    Result of an execution simulation.
    """
    order_id: str
    success: bool
    filled_quantity: int
    average_price: float
    slippage_bps: float
    latency_ms: float
    
    # Detailed breakdown
    market_impact_bps: float = 0.0
    adverse_selection_bps: float = 0.0
    spread_cost_bps: float = 0.0
    
    # Partial fill details
    fill_ratio: float = 1.0
    reason: str = ""


class ExecutionSimulator:
    """
    Simulates realistic order execution.
    
    This class is crucial for accurate backtesting.
    It models:
    - Slippage and market impact
    - Fill probability
    - Adverse selection
    - Execution latency
    """
    
    def __init__(
        self,
        exec_config: ExecutionConfig,
        latency_config: LatencyConfig,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize execution simulator.
        
        Args:
            exec_config: Execution parameters
            latency_config: Latency simulation parameters
            performance_tracker: Optional tracker for recording trades
        """
        self._config = exec_config
        self._latency_config = latency_config
        self._performance_tracker = performance_tracker
        
        # Market impact model
        self._impact_model = MarketImpactModel()
        
        # Latest market data
        self._quotes: Dict[str, Quote] = {}
        self._books: Dict[str, OrderBook] = {}
        
        # Volume estimates (for impact calculation)
        self._avg_daily_volume: Dict[str, int] = {}
        
        # Volatility estimates
        self._volatilities: Dict[str, float] = {}
        
        # Random generator for deterministic simulation
        self._rng = np.random.default_rng(42)
    
    def update_market_data(
        self,
        symbol: str,
        quote: Optional[Quote] = None,
        book: Optional[OrderBook] = None
    ) -> None:
        """Update market data for execution simulation."""
        if quote:
            self._quotes[symbol] = quote
        if book:
            self._books[symbol] = book
    
    def set_volume_estimate(self, symbol: str, avg_daily_volume: int) -> None:
        """Set average daily volume for impact calculation."""
        self._avg_daily_volume[symbol] = avg_daily_volume
    
    def set_volatility(self, symbol: str, volatility: float) -> None:
        """Set volatility estimate for the symbol."""
        self._volatilities[symbol] = volatility
    
    async def execute_order(
        self,
        order: Order,
        router: SmartOrderRouter
    ) -> ExecutionResult:
        """
        Simulate execution of an order.
        
        This is the main entry point for execution simulation.
        
        Args:
            order: Order to execute
            router: Router to update with execution results
            
        Returns:
            ExecutionResult with fill details
        """
        symbol = order.symbol
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        await asyncio.sleep(latency_ms / 1000.0)
        
        # Get current market data
        quote = self._quotes.get(symbol)
        book = self._books.get(symbol)
        
        if not quote and not book:
            # No market data - reject order
            order.status = OrderStatus.REJECTED
            await router.update_order_status(order.order_id, OrderStatus.REJECTED)
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0,
                average_price=0.0,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                reason="No market data available",
            )
        
        # Update order to SENT
        order.status = OrderStatus.SENT
        order.sent_at = datetime.utcnow()
        
        # Dispatch based on order type
        if order.order_type == OrderType.MARKET:
            result = await self._execute_market_order(order, quote, book, latency_ms)
        else:
            result = await self._execute_limit_order(order, quote, book, latency_ms)
        
        # Update router with results
        if result.success:
            status = OrderStatus.FILLED if result.fill_ratio >= 1.0 else OrderStatus.PARTIAL
            await router.update_order_status(
                order.order_id,
                status,
                result.filled_quantity,
                result.average_price
            )
            
            # Record trade in performance tracker
            if self._performance_tracker and result.filled_quantity > 0:
                trade = Trade(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    side=Side.BUY if order.side == OrderSideEnum.BUY else Side.SELL,
                    quantity=result.filled_quantity,
                    price=result.average_price,
                    order_id=order.order_id,
                    fill_id=str(uuid.uuid4()),
                    expected_price=order.expected_price,
                    signal_timestamp=order.created_at,
                    order_timestamp=order.sent_at,
                )
                self._performance_tracker.record_trade(trade)
        
        return result
    
    async def _execute_market_order(
        self,
        order: Order,
        quote: Optional[Quote],
        book: Optional[OrderBook],
        latency_ms: float
    ) -> ExecutionResult:
        """
        Simulate market order execution.
        
        Market orders:
        - Fill immediately at available prices
        - Walk through the book if size exceeds best level
        - Pay the spread plus market impact
        """
        symbol = order.symbol
        
        # Get prices
        if book:
            if order.side == OrderSideEnum.BUY:
                base_price = book.best_ask_price
            else:
                base_price = book.best_bid_price
            spread_bps = book.spread_bps
            mid_price = book.mid_price
        elif quote:
            if order.side == OrderSideEnum.BUY:
                base_price = quote.ask_price
            else:
                base_price = quote.bid_price
            spread_bps = quote.spread_bps
            mid_price = quote.mid_price
        else:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0,
                average_price=0.0,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                reason="No price data",
            )
        
        # Calculate market impact
        volume = self._avg_daily_volume.get(symbol, 1000000)
        volatility = self._volatilities.get(symbol, 0.02)
        
        impact = self._impact_model.estimate_impact(
            side=OrderSide.BID if order.side == OrderSideEnum.BUY else OrderSide.ASK,
            size=order.quantity,
            volatility=volatility,
            avg_daily_volume=volume,
            spread_bps=spread_bps,
        )
        
        # Apply impact to price
        impact_bps = impact["total_impact_bps"]
        
        if order.side == OrderSideEnum.BUY:
            # Buying pushes price up
            fill_price = base_price * (1 + impact_bps / 10000)
        else:
            # Selling pushes price down
            fill_price = base_price * (1 - impact_bps / 10000)
        
        # Add volatility noise
        noise = self._rng.normal(0, self._config.base_slippage_bps) / 10000
        fill_price *= (1 + noise)
        
        # Calculate total slippage from expected (mid) price
        if mid_price > 0:
            if order.side == OrderSideEnum.BUY:
                total_slippage = (fill_price - mid_price) / mid_price * 10000
            else:
                total_slippage = (mid_price - fill_price) / mid_price * 10000
        else:
            total_slippage = 0.0
        
        # Market orders always fill (in liquid markets)
        filled_quantity = order.quantity
        
        logger.debug(
            f"Market order filled: {order.side.value} {filled_quantity} {symbol} "
            f"@ {fill_price:.4f} (slippage={total_slippage:.2f}bps)",
            category=LogCategory.EXECUTION,
            symbol=symbol,
        )
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            filled_quantity=filled_quantity,
            average_price=round(fill_price, 4),
            slippage_bps=total_slippage,
            latency_ms=latency_ms,
            market_impact_bps=impact_bps,
            spread_cost_bps=spread_bps / 2,
            fill_ratio=1.0,
        )
    
    async def _execute_limit_order(
        self,
        order: Order,
        quote: Optional[Quote],
        book: Optional[OrderBook],
        latency_ms: float
    ) -> ExecutionResult:
        """
        Simulate limit order execution.
        
        Limit orders:
        - May or may not fill depending on queue position
        - Subject to adverse selection
        - Better fill price but execution uncertainty
        """
        symbol = order.symbol
        limit_price = order.limit_price
        
        if limit_price is None:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0,
                average_price=0.0,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                reason="Limit order without price",
            )
        
        # Get current market
        if book:
            best_bid = book.best_bid_price
            best_ask = book.best_ask_price
            mid_price = book.mid_price
        elif quote:
            best_bid = quote.bid_price
            best_ask = quote.ask_price
            mid_price = quote.mid_price
        else:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0,
                average_price=0.0,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                reason="No market data",
            )
        
        # Check if order is marketable (crosses spread)
        if order.side == OrderSideEnum.BUY and limit_price >= best_ask:
            # Marketable buy - fills at ask
            return await self._execute_market_order(order, quote, book, latency_ms)
        
        if order.side == OrderSideEnum.SELL and limit_price <= best_bid:
            # Marketable sell - fills at bid
            return await self._execute_market_order(order, quote, book, latency_ms)
        
        # Non-marketable limit order - calculate fill probability
        fill_probability = self._calculate_fill_probability(
            order, mid_price, best_bid, best_ask
        )
        
        # For IOC orders, decide now
        if order.order_type in [OrderType.LIMIT_IOC, OrderType.LIMIT_FOK]:
            # Immediate decision
            if self._rng.random() < fill_probability:
                # Filled at limit price
                fill_price, adverse_selection = self._apply_adverse_selection(
                    order, limit_price, mid_price
                )
                
                if order.order_type == OrderType.LIMIT_FOK:
                    filled_qty = order.quantity
                else:
                    # Partial fill for IOC
                    fill_ratio = self._rng.uniform(0.3, 1.0)
                    filled_qty = int(order.quantity * fill_ratio)
                
                slippage = self._calculate_slippage(order, fill_price, mid_price)
                
                return ExecutionResult(
                    order_id=order.order_id,
                    success=True,
                    filled_quantity=filled_qty,
                    average_price=fill_price,
                    slippage_bps=slippage,
                    latency_ms=latency_ms,
                    adverse_selection_bps=adverse_selection,
                    fill_ratio=filled_qty / order.quantity,
                )
            else:
                # Did not fill
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    filled_quantity=0,
                    average_price=0.0,
                    slippage_bps=0.0,
                    latency_ms=latency_ms,
                    fill_ratio=0.0,
                    reason="Limit order did not execute",
                )
        
        # Regular limit order - simulate over time
        # For simulation simplicity, we use the same logic as IOC
        if self._rng.random() < fill_probability * 1.5:  # Higher prob for patient orders
            fill_price, adverse_selection = self._apply_adverse_selection(
                order, limit_price, mid_price
            )
            
            # Partial fill is common
            fill_ratio = self._rng.uniform(0.5, 1.0) * fill_probability / self._config.limit_order_fill_probability
            fill_ratio = min(fill_ratio, 1.0)
            filled_qty = int(order.quantity * fill_ratio)
            
            if filled_qty == 0:
                filled_qty = min(100, order.quantity)  # At least some fill
            
            slippage = self._calculate_slippage(order, fill_price, mid_price)
            
            return ExecutionResult(
                order_id=order.order_id,
                success=True,
                filled_quantity=filled_qty,
                average_price=fill_price,
                slippage_bps=slippage,
                latency_ms=latency_ms,
                adverse_selection_bps=adverse_selection,
                fill_ratio=fill_ratio,
            )
        else:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0,
                average_price=0.0,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                fill_ratio=0.0,
                reason="Limit order did not execute",
            )
    
    def _calculate_fill_probability(
        self,
        order: Order,
        mid_price: float,
        best_bid: float,
        best_ask: float
    ) -> float:
        """
        Calculate probability of limit order filling.
        
        Factors:
        1. Distance from mid (further = lower probability)
        2. Queue position (estimated as middle of queue)
        3. Volatility (higher vol = better chance)
        """
        limit_price = order.limit_price
        if limit_price is None:
            return 0.0
        
        # Distance from mid in bps
        if mid_price > 0:
            distance_bps = abs(limit_price - mid_price) / mid_price * 10000
        else:
            distance_bps = 0
        
        # Base probability decays with distance
        # Aggressive (inside spread) = higher probability
        # Passive (outside best) = lower probability
        if order.side == OrderSideEnum.BUY:
            if limit_price >= best_bid:
                # At or better than best bid - reasonable chance
                base_prob = self._config.limit_order_fill_probability
            else:
                # Below best bid - lower chance
                base_prob = self._config.limit_order_fill_probability * 0.5
        else:
            if limit_price <= best_ask:
                base_prob = self._config.limit_order_fill_probability
            else:
                base_prob = self._config.limit_order_fill_probability * 0.5
        
        # Queue position adjustment
        queue_position = self._config.average_queue_position
        queue_factor = 1.0 - queue_position * 0.5  # Middle of queue gets ~75% of base prob
        
        # Volatility adjustment
        vol = self._volatilities.get(order.symbol, 0.02)
        vol_factor = min(vol / 0.02, 2.0)  # Higher vol = better fill chance
        
        probability = base_prob * queue_factor * vol_factor
        return min(probability, 0.95)  # Cap at 95%
    
    def _apply_adverse_selection(
        self,
        order: Order,
        limit_price: float,
        mid_price: float
    ) -> tuple:
        """
        Apply adverse selection to fill price.
        
        Adverse selection: You get filled when you don't want to be.
        - If you're buying and price is falling, you're buying something going down
        - If you're selling and price is rising, you're selling something going up
        
        This is a major cost for market makers.
        
        Returns:
            (adjusted_fill_price, adverse_selection_cost_bps)
        """
        base_adverse = self._config.adverse_selection_bps
        
        # Random adverse selection
        # With some probability, the fill is at a worse effective price
        if self._rng.random() < 0.3:  # 30% of fills have adverse selection
            # Adverse movement
            adverse_bps = self._rng.exponential(base_adverse)
            
            if order.side == OrderSideEnum.BUY:
                # After we buy, price drops
                effective_price = limit_price * (1 + adverse_bps / 10000)
            else:
                # After we sell, price rises
                effective_price = limit_price * (1 - adverse_bps / 10000)
            
            return limit_price, adverse_bps  # Fill at limit, but adverse selection cost
        
        return limit_price, 0.0
    
    def _calculate_slippage(
        self,
        order: Order,
        fill_price: float,
        mid_price: float
    ) -> float:
        """Calculate slippage from mid price in basis points."""
        if mid_price <= 0:
            return 0.0
        
        if order.side == OrderSideEnum.BUY:
            return (fill_price - mid_price) / mid_price * 10000
        else:
            return (mid_price - fill_price) / mid_price * 10000
    
    def _simulate_latency(self) -> float:
        """Simulate execution latency."""
        base = self._latency_config.base_latency_ms
        jitter = self._rng.normal(
            self._latency_config.jitter_mean_ms,
            self._latency_config.jitter_std_ms
        )
        
        # Occasional spikes
        if self._rng.random() < self._latency_config.spike_probability:
            spike = base * self._latency_config.spike_multiplier
            return max(0, base + jitter + spike)
        
        return max(0, base + jitter)
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic simulation."""
        self._rng = np.random.default_rng(seed)
