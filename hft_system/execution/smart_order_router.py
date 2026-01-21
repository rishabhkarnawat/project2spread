"""
Smart Order Router (SOR) for HFT Research System
=================================================

The Smart Order Router decides HOW to execute a trade:
- Market vs limit order
- Order size and timing
- Price placement for limit orders
- Split across venues (in production)

SMART ORDER ROUTING IN REAL HFT:
================================

Real SORs make complex decisions:

1. VENUE SELECTION:
   - Route to venue with best price (NBBO)
   - Consider hidden liquidity (dark pools)
   - Factor in rebates vs fees
   - Avoid adverse selection on certain venues

2. ORDER TYPE SELECTION:
   - Market: Guaranteed fill, but crosses spread
   - Limit: Better price, but may not fill
   - IOC (Immediate or Cancel): Limit with no queue wait
   - FOK (Fill or Kill): All or nothing
   - Pegged: Tracks bid/offer automatically

3. TIMING:
   - Immediate: High urgency signals
   - TWAP: Spread over time to reduce impact
   - VWAP: Follow volume pattern
   - Opportunistic: Wait for favorable conditions

4. SIZE:
   - Based on signal confidence
   - Limited by risk constraints
   - Scaled by liquidity available

OUR IMPLEMENTATION:
==================
Simplified single-venue routing focused on:
- Market vs limit decision
- Limit price placement
- Order sizing
- Urgency handling
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable

import numpy as np

from ..data.orderbook_simulator import OrderBook, OrderSide
from ..data.market_data_feed import Quote
from ..signals.signal_aggregator import CombinedSignal
from ..infra.config import ExecutionConfig
from ..infra.logging import get_logger, LogCategory


logger = get_logger()


class OrderType(Enum):
    """Order types supported by the router."""
    MARKET = "MARKET"              # Execute immediately at best available
    LIMIT = "LIMIT"                # Execute at specified price or better
    LIMIT_IOC = "LIMIT_IOC"        # Immediate or cancel
    LIMIT_FOK = "LIMIT_FOK"        # Fill or kill


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"            # Created, not yet sent
    SENT = "SENT"                  # Sent to exchange
    PARTIAL = "PARTIAL"            # Partially filled
    FILLED = "FILLED"              # Completely filled
    CANCELLED = "CANCELLED"        # Cancelled before full fill
    REJECTED = "REJECTED"          # Rejected by risk or exchange


class OrderSideEnum(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """
    Represents a trading order.
    
    In production, orders would include:
    - Exchange-specific fields
    - Regulatory identifiers
    - Microsecond timestamps
    - Routing instructions
    """
    order_id: str
    symbol: str
    side: OrderSideEnum
    quantity: int
    order_type: OrderType
    
    # Price (for limit orders)
    limit_price: Optional[float] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Signal that triggered this order
    signal_value: float = 0.0
    signal_confidence: float = 0.0
    
    # Expected prices for slippage tracking
    expected_price: float = 0.0
    
    # Fill details
    fills: List[Dict] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED, 
            OrderStatus.CANCELLED, 
            OrderStatus.REJECTED
        ]
    
    @property
    def slippage_bps(self) -> float:
        """Calculate slippage from expected price."""
        if self.expected_price <= 0 or self.average_fill_price <= 0:
            return 0.0
        
        if self.side == OrderSideEnum.BUY:
            # For buys, positive slippage = paid more than expected
            return (self.average_fill_price - self.expected_price) / self.expected_price * 10000
        else:
            # For sells, positive slippage = received less than expected
            return (self.expected_price - self.average_fill_price) / self.expected_price * 10000
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "slippage_bps": self.slippage_bps,
        }


# Callback for order updates
OrderCallback = Callable[[Order], Awaitable[None]]


class SmartOrderRouter:
    """
    Smart Order Router for intelligent order execution.
    
    Makes decisions about:
    - Order type (market vs limit)
    - Limit price placement
    - Order sizing
    - Timing and urgency
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize the smart order router.
        
        Args:
            config: Execution configuration
        """
        self._config = config
        
        # Pending orders by symbol
        self._pending_orders: Dict[str, List[Order]] = {}
        
        # All orders (for tracking)
        self._all_orders: Dict[str, Order] = {}
        
        # Callbacks for order events
        self._callbacks: List[OrderCallback] = []
        
        # Latest market data
        self._latest_quotes: Dict[str, Quote] = {}
        self._latest_books: Dict[str, OrderBook] = {}
    
    def update_market_data(
        self,
        symbol: str,
        quote: Optional[Quote] = None,
        book: Optional[OrderBook] = None
    ) -> None:
        """Update market data for routing decisions."""
        if quote:
            self._latest_quotes[symbol] = quote
        if book:
            self._latest_books[symbol] = book
    
    def register_callback(self, callback: OrderCallback) -> None:
        """Register callback for order updates."""
        self._callbacks.append(callback)
    
    async def route_signal(
        self,
        signal: CombinedSignal,
        max_position: int
    ) -> Optional[Order]:
        """
        Route a trading signal to an order.
        
        This is the main entry point for converting signals to orders.
        
        Args:
            signal: Combined trading signal
            max_position: Maximum position size allowed
            
        Returns:
            Created order, or None if signal doesn't warrant trading
        """
        # Check if signal warrants trading
        if not signal.should_trade():
            return None
        
        symbol = signal.symbol
        
        # Get market data
        quote = self._latest_quotes.get(symbol)
        book = self._latest_books.get(symbol)
        
        if not quote and not book:
            logger.warning(
                f"No market data for {symbol}, cannot route order",
                category=LogCategory.EXECUTION,
                symbol=symbol
            )
            return None
        
        # Determine order parameters
        side = OrderSideEnum.BUY if signal.suggested_side == "BUY" else OrderSideEnum.SELL
        
        # Calculate order size
        size = self._calculate_order_size(signal, max_position)
        if size <= 0:
            return None
        
        # Determine order type
        order_type, limit_price = self._determine_order_type(
            signal, side, quote, book
        )
        
        # Get expected price
        if book:
            mid_price = book.mid_price
        elif quote:
            mid_price = quote.mid_price
        else:
            mid_price = 0.0
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=size,
            order_type=order_type,
            limit_price=limit_price,
            signal_value=signal.signal_value,
            signal_confidence=signal.confidence,
            expected_price=mid_price,
        )
        
        # Store order
        self._all_orders[order.order_id] = order
        if symbol not in self._pending_orders:
            self._pending_orders[symbol] = []
        self._pending_orders[symbol].append(order)
        
        logger.info(
            f"Routed order: {side.value} {size} {symbol} @ {order_type.value} "
            f"(signal={signal.signal_value:.3f}, conf={signal.confidence:.2f})",
            category=LogCategory.EXECUTION,
            symbol=symbol,
            order_id=order.order_id,
        )
        
        return order
    
    def _calculate_order_size(
        self,
        signal: CombinedSignal,
        max_position: int
    ) -> int:
        """
        Calculate order size based on signal and constraints.
        
        Size is determined by:
        1. Signal confidence (higher = larger)
        2. Signal strength (stronger = larger)
        3. Spread (wider = smaller to reduce cost)
        4. Max position limit
        """
        # Base size from signal
        base_size = signal.suggested_size_pct * max_position
        
        # Scale by confidence
        confidence_scale = signal.confidence ** 2  # Quadratic for conservatism
        
        # Reduce for wide spreads
        spread_scale = 1.0
        if signal.spread_bps > 10:
            spread_scale = 0.8
        elif signal.spread_bps > 20:
            spread_scale = 0.5
        elif signal.spread_bps > 50:
            spread_scale = 0.2
        
        # Calculate final size
        size = int(base_size * confidence_scale * spread_scale)
        
        # Apply constraints
        size = max(size, 0)
        size = min(size, self._config.max_order_size)
        size = min(size, max_position)
        
        # Round to lot size (typically 100 for US equities)
        size = (size // 100) * 100
        
        return size
    
    def _determine_order_type(
        self,
        signal: CombinedSignal,
        side: OrderSideEnum,
        quote: Optional[Quote],
        book: Optional[OrderBook]
    ) -> tuple:
        """
        Determine whether to use market or limit order.
        
        Decision factors:
        1. Signal urgency
        2. Spread width
        3. Signal confidence
        
        Returns:
            (order_type, limit_price or None)
        """
        # Get spread
        if book:
            spread_bps = book.spread_bps
            best_bid = book.best_bid_price
            best_ask = book.best_ask_price
            mid_price = book.mid_price
        elif quote:
            spread_bps = quote.spread_bps
            best_bid = quote.bid_price
            best_ask = quote.ask_price
            mid_price = quote.mid_price
        else:
            return OrderType.MARKET, None
        
        # HIGH urgency or very high confidence -> Market order
        if signal.urgency == "HIGH" or signal.confidence > 0.9:
            return OrderType.MARKET, None
        
        # Wide spread -> Limit order (don't pay full spread)
        if spread_bps > 5:
            # Place limit at aggressive price (inside the spread)
            if side == OrderSideEnum.BUY:
                # Buy: bid up slightly from mid
                limit_price = mid_price - (spread_bps / 10000 * mid_price * 0.25)
                limit_price = max(limit_price, best_bid)  # Don't go below bid
            else:
                # Sell: offer down slightly from mid
                limit_price = mid_price + (spread_bps / 10000 * mid_price * 0.25)
                limit_price = min(limit_price, best_ask)  # Don't go above ask
            
            # Use IOC for time-sensitive signals
            if signal.urgency == "NORMAL":
                return OrderType.LIMIT_IOC, round(limit_price, 4)
            else:
                return OrderType.LIMIT, round(limit_price, 4)
        
        # Tight spread -> Market order (crossing is cheap)
        return OrderType.MARKET, None
    
    def create_order(
        self,
        symbol: str,
        side: OrderSideEnum,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> Order:
        """
        Create an order directly (not from signal).
        
        Used for:
        - Manual orders
        - Risk management orders
        - Closing positions
        """
        quote = self._latest_quotes.get(symbol)
        expected_price = quote.mid_price if quote else 0.0
        
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            expected_price=expected_price,
        )
        
        self._all_orders[order.order_id] = order
        if symbol not in self._pending_orders:
            self._pending_orders[symbol] = []
        self._pending_orders[symbol].append(order)
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Returns True if cancellation was successful.
        """
        order = self._all_orders.get(order_id)
        if not order:
            return False
        
        if order.is_complete:
            return False
        
        order.status = OrderStatus.CANCELLED
        
        # Notify callbacks
        for callback in self._callbacks:
            await callback(order)
        
        logger.info(
            f"Cancelled order {order_id}",
            category=LogCategory.EXECUTION,
            order_id=order_id,
        )
        
        return True
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all pending orders, optionally for a specific symbol.
        
        Returns number of orders cancelled.
        """
        cancelled = 0
        
        symbols = [symbol] if symbol else list(self._pending_orders.keys())
        
        for sym in symbols:
            orders = self._pending_orders.get(sym, [])
            for order in orders:
                if not order.is_complete:
                    if await self.cancel_order(order.order_id):
                        cancelled += 1
        
        return cancelled
    
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_qty: int = 0,
        fill_price: float = 0.0
    ) -> None:
        """
        Update order status (called by execution simulator).
        """
        order = self._all_orders.get(order_id)
        if not order:
            return
        
        order.status = status
        
        if filled_qty > 0:
            # Update average fill price
            old_value = order.filled_quantity * order.average_fill_price
            new_value = filled_qty * fill_price
            order.filled_quantity += filled_qty
            order.average_fill_price = (old_value + new_value) / order.filled_quantity
            
            # Record fill
            order.fills.append({
                "quantity": filled_qty,
                "price": fill_price,
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()
        
        # Notify callbacks
        for callback in self._callbacks:
            await callback(order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self._all_orders.get(order_id)
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get pending orders."""
        if symbol:
            return [o for o in self._pending_orders.get(symbol, []) if not o.is_complete]
        
        all_pending = []
        for orders in self._pending_orders.values():
            all_pending.extend([o for o in orders if not o.is_complete])
        return all_pending
    
    def get_execution_statistics(self) -> Dict[str, float]:
        """
        Get execution quality statistics.
        
        Key metrics for execution quality:
        - Fill rate
        - Average slippage
        - Market vs limit ratio
        """
        filled_orders = [o for o in self._all_orders.values() if o.status == OrderStatus.FILLED]
        all_orders = list(self._all_orders.values())
        
        if not all_orders:
            return {}
        
        # Fill rate
        fill_rate = len(filled_orders) / len(all_orders)
        
        # Slippage
        slippages = [o.slippage_bps for o in filled_orders if o.slippage_bps != 0]
        avg_slippage = np.mean(slippages) if slippages else 0.0
        
        # Order type distribution
        market_orders = sum(1 for o in all_orders if o.order_type == OrderType.MARKET)
        market_ratio = market_orders / len(all_orders)
        
        return {
            "total_orders": len(all_orders),
            "filled_orders": len(filled_orders),
            "fill_rate": fill_rate,
            "avg_slippage_bps": avg_slippage,
            "market_order_ratio": market_ratio,
        }
