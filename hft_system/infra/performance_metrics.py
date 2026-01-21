"""
Performance Metrics for HFT Research System
============================================

This module tracks and computes trading performance metrics:
- PnL tracking (realized, unrealized, total)
- Risk-adjusted returns (Sharpe, Sortino)
- Execution quality metrics
- Inventory and risk metrics

METRICS THAT MATTER IN HFT:
===========================

1. PROFITABILITY:
   - Gross PnL: Raw profit before costs
   - Net PnL: After transaction costs, rebates
   - PnL per trade: Efficiency measure
   - Hit rate: % of profitable trades

2. RISK-ADJUSTED:
   - Sharpe ratio: Return per unit of risk
   - Sortino ratio: Return per unit of downside risk
   - Max drawdown: Worst peak-to-trough decline
   - Calmar ratio: Return / Max drawdown

3. EXECUTION QUALITY:
   - Slippage: Difference from expected fill price
   - Fill rate: % of orders that execute
   - Latency distribution: Time from signal to fill

4. INVENTORY RISK:
   - Average inventory: Exposure over time
   - Inventory half-life: How quickly positions unwind
   - Adverse selection: Losses from informed traders

These metrics are computed in real-time and aggregated
for post-trade analysis.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque, Tuple
from enum import Enum
import threading


class Side(Enum):
    """Trade side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """
    Represents a single executed trade.
    
    In production, would include:
    - Microsecond timestamps
    - Venue information
    - Fees and rebates
    - Counterparty info (if available)
    """
    timestamp: datetime
    symbol: str
    side: Side
    quantity: int
    price: float
    order_id: str
    fill_id: str
    
    # Execution quality fields
    expected_price: float = 0.0  # What we expected to get
    signal_timestamp: Optional[datetime] = None  # When signal was generated
    order_timestamp: Optional[datetime] = None   # When order was sent
    
    # Fees (simplified)
    fees: float = 0.0
    rebate: float = 0.0
    
    @property
    def signed_quantity(self) -> int:
        """Positive for buys, negative for sells."""
        return self.quantity if self.side == Side.BUY else -self.quantity
    
    @property
    def notional(self) -> float:
        """Dollar value of the trade."""
        return self.quantity * self.price
    
    @property
    def slippage_bps(self) -> float:
        """
        Slippage in basis points.
        Positive = worse than expected, negative = better.
        """
        if self.expected_price <= 0:
            return 0.0
        if self.side == Side.BUY:
            return (self.price - self.expected_price) / self.expected_price * 10000
        else:
            return (self.expected_price - self.price) / self.expected_price * 10000
    
    @property
    def latency_us(self) -> Optional[float]:
        """Latency from signal to fill in microseconds."""
        if self.signal_timestamp and self.timestamp:
            delta = self.timestamp - self.signal_timestamp
            return delta.total_seconds() * 1_000_000
        return None


@dataclass
class Position:
    """
    Tracks position in a single symbol.
    
    Uses FIFO cost basis for PnL calculation.
    """
    symbol: str
    quantity: int = 0
    cost_basis: float = 0.0  # Total cost of position
    realized_pnl: float = 0.0
    
    # For tracking
    entry_prices: Deque[Tuple[int, float]] = field(default_factory=deque)
    
    @property
    def average_price(self) -> float:
        """Average entry price."""
        if self.quantity == 0:
            return 0.0
        return self.cost_basis / abs(self.quantity)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL at current market price."""
        if self.quantity == 0:
            return 0.0
        return self.quantity * (current_price - self.average_price)
    
    def update(self, trade: Trade) -> float:
        """
        Update position with a new trade.
        Returns realized PnL from this trade.
        """
        qty = trade.signed_quantity
        realized = 0.0
        
        if self.quantity == 0:
            # Opening new position
            self.quantity = qty
            self.cost_basis = abs(qty) * trade.price
            self.entry_prices.append((abs(qty), trade.price))
            
        elif (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            # Adding to position
            self.quantity += qty
            self.cost_basis += abs(qty) * trade.price
            self.entry_prices.append((abs(qty), trade.price))
            
        else:
            # Reducing/closing/flipping position
            close_qty = min(abs(self.quantity), abs(qty))
            
            # Calculate realized PnL (FIFO)
            remaining_to_close = close_qty
            while remaining_to_close > 0 and self.entry_prices:
                entry_qty, entry_price = self.entry_prices[0]
                closed_from_entry = min(entry_qty, remaining_to_close)
                
                if self.quantity > 0:
                    # Was long, selling
                    realized += closed_from_entry * (trade.price - entry_price)
                else:
                    # Was short, buying
                    realized += closed_from_entry * (entry_price - trade.price)
                
                remaining_to_close -= closed_from_entry
                
                if closed_from_entry >= entry_qty:
                    self.entry_prices.popleft()
                else:
                    self.entry_prices[0] = (entry_qty - closed_from_entry, entry_price)
            
            # Update position
            new_qty = self.quantity + qty
            
            if new_qty == 0:
                self.cost_basis = 0.0
            elif (new_qty > 0 and self.quantity > 0) or (new_qty < 0 and self.quantity < 0):
                # Partial close
                self.cost_basis = self.cost_basis * (abs(new_qty) / abs(self.quantity))
            else:
                # Flipped position
                flip_qty = abs(qty) - close_qty
                self.cost_basis = flip_qty * trade.price
                self.entry_prices.clear()
                self.entry_prices.append((flip_qty, trade.price))
            
            self.quantity = new_qty
        
        self.realized_pnl += realized
        return realized


class PerformanceTracker:
    """
    Comprehensive performance tracking for the trading system.
    
    Thread-safe for use across async components.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance tracker.
        
        Args:
            risk_free_rate: Annualized risk-free rate for Sharpe calculation
        """
        self._risk_free_rate = risk_free_rate
        self._lock = threading.Lock()
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        self._current_prices: Dict[str, float] = {}
        
        # Trade history
        self._trades: List[Trade] = []
        
        # PnL tracking (for time series)
        self._pnl_history: Deque[Tuple[datetime, float]] = deque(maxlen=100000)
        self._returns: Deque[float] = deque(maxlen=100000)
        
        # High-water mark for drawdown
        self._high_water_mark: float = 0.0
        self._max_drawdown: float = 0.0
        
        # Execution metrics
        self._slippage_samples: Deque[float] = deque(maxlen=10000)
        self._latency_samples: Deque[float] = deque(maxlen=10000)
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update current market price for a symbol."""
        with self._lock:
            self._current_prices[symbol] = price
    
    def record_trade(self, trade: Trade) -> Dict[str, float]:
        """
        Record a trade and update all metrics.
        
        Returns dict with immediate metrics from this trade.
        """
        with self._lock:
            # Initialize position if needed
            if trade.symbol not in self._positions:
                self._positions[trade.symbol] = Position(symbol=trade.symbol)
            
            # Update position and get realized PnL
            realized_pnl = self._positions[trade.symbol].update(trade)
            
            # Store trade
            self._trades.append(trade)
            
            # Update execution metrics
            if trade.slippage_bps != 0:
                self._slippage_samples.append(trade.slippage_bps)
            if trade.latency_us is not None:
                self._latency_samples.append(trade.latency_us)
            
            # Update PnL history
            total_pnl = self.total_pnl_unlocked()
            self._pnl_history.append((trade.timestamp, total_pnl))
            
            # Update returns (if we have history)
            if len(self._pnl_history) > 1:
                prev_pnl = self._pnl_history[-2][1]
                if prev_pnl != 0:
                    ret = (total_pnl - prev_pnl) / abs(prev_pnl) if prev_pnl != 0 else 0
                    self._returns.append(ret)
            
            # Update drawdown tracking
            if total_pnl > self._high_water_mark:
                self._high_water_mark = total_pnl
            elif self._high_water_mark > 0:
                drawdown = (self._high_water_mark - total_pnl) / self._high_water_mark
                self._max_drawdown = max(self._max_drawdown, drawdown)
            
            return {
                "realized_pnl": realized_pnl,
                "total_pnl": total_pnl,
                "slippage_bps": trade.slippage_bps,
                "position": self._positions[trade.symbol].quantity
            }
    
    def total_pnl_unlocked(self) -> float:
        """Calculate total PnL (call with lock held)."""
        realized = sum(p.realized_pnl for p in self._positions.values())
        unrealized = sum(
            p.unrealized_pnl(self._current_prices.get(p.symbol, p.average_price))
            for p in self._positions.values()
        )
        return realized + unrealized
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position in a symbol."""
        with self._lock:
            return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        with self._lock:
            return dict(self._positions)
    
    def get_gross_position(self) -> int:
        """Get gross position (sum of absolute values)."""
        with self._lock:
            return sum(abs(p.quantity) for p in self._positions.values())
    
    def get_net_position(self) -> int:
        """Get net position (can be negative)."""
        with self._lock:
            return sum(p.quantity for p in self._positions.values())
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        This is the key output for strategy evaluation.
        """
        with self._lock:
            total_pnl = self.total_pnl_unlocked()
            realized_pnl = sum(p.realized_pnl for p in self._positions.values())
            unrealized_pnl = total_pnl - realized_pnl
            
            # Trade statistics
            num_trades = len(self._trades)
            if num_trades > 0:
                winning_trades = sum(1 for t in self._trades if self._get_trade_pnl(t) > 0)
                hit_rate = winning_trades / num_trades
                avg_trade_pnl = total_pnl / num_trades
            else:
                hit_rate = 0.0
                avg_trade_pnl = 0.0
            
            # Risk metrics
            returns = list(self._returns)
            if len(returns) > 1:
                returns_array = np.array(returns)
                sharpe = self._calculate_sharpe(returns_array)
                sortino = self._calculate_sortino(returns_array)
                volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
                sortino = 0.0
                volatility = 0.0
            
            # Execution quality
            if self._slippage_samples:
                avg_slippage = np.mean(list(self._slippage_samples))
            else:
                avg_slippage = 0.0
            
            if self._latency_samples:
                avg_latency = np.mean(list(self._latency_samples))
                p99_latency = np.percentile(list(self._latency_samples), 99)
            else:
                avg_latency = 0.0
                p99_latency = 0.0
            
            return {
                # PnL
                "total_pnl": total_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                
                # Trade stats
                "num_trades": num_trades,
                "hit_rate": hit_rate,
                "avg_trade_pnl": avg_trade_pnl,
                
                # Risk metrics
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "volatility": volatility,
                "max_drawdown": self._max_drawdown,
                
                # Position
                "gross_position": self.get_gross_position(),
                "net_position": self.get_net_position(),
                
                # Execution
                "avg_slippage_bps": avg_slippage,
                "avg_latency_us": avg_latency,
                "p99_latency_us": p99_latency,
            }
    
    def _get_trade_pnl(self, trade: Trade) -> float:
        """Estimate PnL contribution of a single trade."""
        # Simplified - in reality would need to track entry/exit pairs
        return 0.0  # Placeholder
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (E[R] - Rf) / std(R)
        
        Annualized assuming returns are per-trade.
        In HFT, this is often calculated intraday.
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - self._risk_free_rate / 252  # Daily rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio.
        
        Like Sharpe but only penalizes downside volatility.
        More appropriate for asymmetric return distributions.
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self._risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def get_pnl_series(self) -> List[Tuple[datetime, float]]:
        """Get PnL time series for plotting."""
        with self._lock:
            return list(self._pnl_history)
    
    def reset(self) -> None:
        """Reset all metrics (e.g., for new backtest run)."""
        with self._lock:
            self._positions.clear()
            self._current_prices.clear()
            self._trades.clear()
            self._pnl_history.clear()
            self._returns.clear()
            self._high_water_mark = 0.0
            self._max_drawdown = 0.0
            self._slippage_samples.clear()
            self._latency_samples.clear()


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a readable report.
    
    This is what you'd show to PMs and risk managers.
    """
    report = """
╔════════════════════════════════════════════════════════════╗
║                 TRADING PERFORMANCE REPORT                 ║
╠════════════════════════════════════════════════════════════╣
║ PNL METRICS                                                ║
║   Total PnL:        ${total_pnl:>12,.2f}                         ║
║   Realized PnL:     ${realized_pnl:>12,.2f}                         ║
║   Unrealized PnL:   ${unrealized_pnl:>12,.2f}                         ║
╠════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                           ║
║   Number of Trades: {num_trades:>12,}                         ║
║   Hit Rate:         {hit_rate:>12.1%}                         ║
║   Avg Trade PnL:    ${avg_trade_pnl:>12,.2f}                         ║
╠════════════════════════════════════════════════════════════╣
║ RISK METRICS                                               ║
║   Sharpe Ratio:     {sharpe_ratio:>12.2f}                         ║
║   Sortino Ratio:    {sortino_ratio:>12.2f}                         ║
║   Volatility:       {volatility:>12.2%}                         ║
║   Max Drawdown:     {max_drawdown:>12.2%}                         ║
╠════════════════════════════════════════════════════════════╣
║ POSITION                                                   ║
║   Gross Position:   {gross_position:>12,}                         ║
║   Net Position:     {net_position:>12,}                         ║
╠════════════════════════════════════════════════════════════╣
║ EXECUTION QUALITY                                          ║
║   Avg Slippage:     {avg_slippage_bps:>12.2f} bps                    ║
║   Avg Latency:      {avg_latency_us:>12.1f} μs                     ║
║   P99 Latency:      {p99_latency_us:>12.1f} μs                     ║
╚════════════════════════════════════════════════════════════╝
""".format(**metrics)
    return report
