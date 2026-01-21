"""
Position Limits for HFT Research System
========================================

Position limits are CRITICAL risk controls in trading:
- Prevent excessive exposure to single names
- Control overall portfolio risk
- Comply with regulatory requirements
- Protect firm capital

REAL-WORLD POSITION LIMITS:
===========================

1. REGULATORY LIMITS:
   - Large trader reporting (13H): > 2M shares or $20M
   - Position limits for derivatives
   - Aggregation across accounts/entities

2. EXCHANGE LIMITS:
   - Position limits per contract
   - Accountability levels
   - Reportable positions

3. INTERNAL LIMITS:
   - Per-strategy limits
   - Per-trader limits
   - Firm-wide exposure limits
   - Concentration limits

4. RISK-BASED LIMITS:
   - VaR limits
   - Greeks limits (delta, gamma, vega)
   - Stress test limits

IMPLEMENTATION:
==============
This module implements:
- Per-symbol position limits
- Gross/net position limits
- Real-time limit checking
- Limit breach handling
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from enum import Enum
import threading

from ..infra.config import RiskConfig
from ..infra.logging import get_logger, LogCategory
from ..infra.performance_metrics import Position, PerformanceTracker


logger = get_logger()


class LimitType(Enum):
    """Types of position limits."""
    SYMBOL = "symbol"              # Per-symbol limit
    GROSS = "gross"                # Total absolute position
    NET = "net"                    # Net exposure (long - short)
    SECTOR = "sector"              # Sector concentration
    NOTIONAL = "notional"          # Dollar exposure


class LimitAction(Enum):
    """Actions when limit is breached."""
    WARN = "warn"                  # Log warning, allow trade
    BLOCK = "block"                # Reject order
    REDUCE = "reduce"              # Reduce order size to fit
    FLATTEN = "flatten"            # Flatten entire position


@dataclass
class LimitBreach:
    """Record of a limit breach event."""
    timestamp: datetime
    limit_type: LimitType
    symbol: Optional[str]
    current_value: float
    limit_value: float
    requested_change: float
    action_taken: LimitAction
    order_id: Optional[str] = None
    
    @property
    def breach_pct(self) -> float:
        """How much the limit was exceeded (percentage)."""
        if self.limit_value == 0:
            return 0.0
        return (self.current_value - self.limit_value) / abs(self.limit_value) * 100


@dataclass
class PositionLimitConfig:
    """Configuration for a specific limit."""
    limit_type: LimitType
    limit_value: float
    action: LimitAction = LimitAction.BLOCK
    symbol: Optional[str] = None    # For symbol-specific limits
    warning_threshold: float = 0.8  # Warn at 80% of limit


class PositionLimitManager:
    """
    Manages position limits and enforces risk constraints.
    
    This is a critical component that must:
    - Be extremely fast (sub-microsecond in production)
    - Never allow limit breaches
    - Fail safe (block if uncertain)
    """
    
    def __init__(
        self,
        config: RiskConfig,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize position limit manager.
        
        Args:
            config: Risk configuration with limit values
            performance_tracker: Tracker for current positions
        """
        self._config = config
        self._tracker = performance_tracker
        
        # Position state
        self._positions: Dict[str, int] = {}
        
        # Limit configurations
        self._limits: List[PositionLimitConfig] = []
        
        # Breach history
        self._breaches: List[LimitBreach] = []
        
        # Blocked symbols (emergency)
        self._blocked_symbols: Set[str] = set()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default limits from config
        self._setup_default_limits()
    
    def _setup_default_limits(self) -> None:
        """Set up default position limits from config."""
        # Per-symbol limit
        self._limits.append(PositionLimitConfig(
            limit_type=LimitType.SYMBOL,
            limit_value=self._config.max_position_per_symbol,
            action=LimitAction.BLOCK,
        ))
        
        # Gross position limit
        self._limits.append(PositionLimitConfig(
            limit_type=LimitType.GROSS,
            limit_value=self._config.max_gross_position,
            action=LimitAction.BLOCK,
        ))
        
        # Net position limit
        self._limits.append(PositionLimitConfig(
            limit_type=LimitType.NET,
            limit_value=self._config.max_net_position,
            action=LimitAction.BLOCK,
        ))
    
    def update_position(self, symbol: str, quantity: int) -> None:
        """
        Update tracked position.
        
        Called when trades execute.
        """
        with self._lock:
            self._positions[symbol] = self._positions.get(symbol, 0) + quantity
    
    def set_position(self, symbol: str, quantity: int) -> None:
        """Set position directly (e.g., from external source)."""
        with self._lock:
            self._positions[symbol] = quantity
    
    def get_position(self, symbol: str) -> int:
        """Get current position in a symbol."""
        with self._lock:
            return self._positions.get(symbol, 0)
    
    def get_all_positions(self) -> Dict[str, int]:
        """Get all positions."""
        with self._lock:
            return dict(self._positions)
    
    def check_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        order_id: Optional[str] = None
    ) -> tuple:
        """
        Check if an order would breach position limits.
        
        This is called BEFORE sending an order.
        Must be extremely fast.
        
        Args:
            symbol: Symbol to trade
            side: "BUY" or "SELL"
            quantity: Order quantity
            order_id: Optional order ID for tracking
            
        Returns:
            (allowed: bool, adjusted_quantity: int, breaches: List[LimitBreach])
        """
        with self._lock:
            # Check if symbol is blocked
            if symbol in self._blocked_symbols:
                breach = LimitBreach(
                    timestamp=datetime.utcnow(),
                    limit_type=LimitType.SYMBOL,
                    symbol=symbol,
                    current_value=0,
                    limit_value=0,
                    requested_change=quantity,
                    action_taken=LimitAction.BLOCK,
                    order_id=order_id,
                )
                return False, 0, [breach]
            
            # Calculate resulting position
            current_position = self._positions.get(symbol, 0)
            signed_qty = quantity if side == "BUY" else -quantity
            new_position = current_position + signed_qty
            
            # Calculate gross and net
            new_gross = self._calculate_gross(symbol, new_position)
            new_net = self._calculate_net(symbol, new_position)
            
            breaches = []
            allowed = True
            adjusted_qty = quantity
            
            # Check each limit
            for limit in self._limits:
                breach = self._check_single_limit(
                    limit, symbol, new_position, new_gross, new_net,
                    signed_qty, order_id
                )
                
                if breach:
                    breaches.append(breach)
                    self._breaches.append(breach)
                    
                    if breach.action_taken == LimitAction.BLOCK:
                        allowed = False
                        adjusted_qty = 0
                    elif breach.action_taken == LimitAction.REDUCE:
                        # Calculate maximum allowed
                        adjusted_qty = self._calculate_max_allowed(
                            limit, symbol, side, current_position
                        )
                        allowed = adjusted_qty > 0
            
            # Log warnings even if allowed
            for breach in breaches:
                if breach.action_taken == LimitAction.WARN:
                    logger.log_risk_event(
                        "LIMIT_WARNING",
                        f"{breach.limit_type.value} limit warning for {symbol}: "
                        f"{breach.current_value}/{breach.limit_value}",
                        severity="WARNING",
                        symbol=symbol,
                    )
                elif breach.action_taken == LimitAction.BLOCK:
                    logger.log_risk_event(
                        "LIMIT_BREACH",
                        f"{breach.limit_type.value} limit BLOCKED for {symbol}: "
                        f"{breach.current_value}/{breach.limit_value}",
                        severity="CRITICAL",
                        symbol=symbol,
                    )
            
            return allowed, adjusted_qty, breaches
    
    def _check_single_limit(
        self,
        limit: PositionLimitConfig,
        symbol: str,
        new_position: int,
        new_gross: int,
        new_net: int,
        requested_change: int,
        order_id: Optional[str]
    ) -> Optional[LimitBreach]:
        """Check a single limit configuration."""
        # Determine which value to check
        if limit.limit_type == LimitType.SYMBOL:
            if limit.symbol and limit.symbol != symbol:
                return None  # This limit doesn't apply
            current_value = abs(new_position)
        elif limit.limit_type == LimitType.GROSS:
            current_value = new_gross
        elif limit.limit_type == LimitType.NET:
            current_value = abs(new_net)
        else:
            return None
        
        limit_value = limit.limit_value
        
        # Check breach
        if current_value > limit_value:
            return LimitBreach(
                timestamp=datetime.utcnow(),
                limit_type=limit.limit_type,
                symbol=symbol if limit.limit_type == LimitType.SYMBOL else None,
                current_value=current_value,
                limit_value=limit_value,
                requested_change=requested_change,
                action_taken=limit.action,
                order_id=order_id,
            )
        
        # Check warning threshold
        if current_value > limit_value * limit.warning_threshold:
            return LimitBreach(
                timestamp=datetime.utcnow(),
                limit_type=limit.limit_type,
                symbol=symbol if limit.limit_type == LimitType.SYMBOL else None,
                current_value=current_value,
                limit_value=limit_value,
                requested_change=requested_change,
                action_taken=LimitAction.WARN,
                order_id=order_id,
            )
        
        return None
    
    def _calculate_gross(self, symbol: str, new_position: int) -> int:
        """Calculate gross position after update."""
        gross = 0
        for sym, pos in self._positions.items():
            if sym == symbol:
                gross += abs(new_position)
            else:
                gross += abs(pos)
        if symbol not in self._positions:
            gross += abs(new_position)
        return gross
    
    def _calculate_net(self, symbol: str, new_position: int) -> int:
        """Calculate net position after update."""
        net = 0
        for sym, pos in self._positions.items():
            if sym == symbol:
                net += new_position
            else:
                net += pos
        if symbol not in self._positions:
            net += new_position
        return net
    
    def _calculate_max_allowed(
        self,
        limit: PositionLimitConfig,
        symbol: str,
        side: str,
        current_position: int
    ) -> int:
        """Calculate maximum order size that fits within limit."""
        if limit.limit_type == LimitType.SYMBOL:
            # Max we can add without exceeding limit
            current_abs = abs(current_position)
            remaining = max(0, limit.limit_value - current_abs)
            return int(remaining)
        
        elif limit.limit_type == LimitType.GROSS:
            current_gross = sum(abs(p) for p in self._positions.values())
            remaining = max(0, limit.limit_value - current_gross)
            return int(remaining)
        
        return 0
    
    def block_symbol(self, symbol: str, reason: str = "") -> None:
        """
        Block all trading in a symbol.
        
        Used for:
        - Detected anomalies
        - News events
        - Risk limit breaches
        """
        with self._lock:
            self._blocked_symbols.add(symbol)
        
        logger.log_risk_event(
            "SYMBOL_BLOCKED",
            f"Trading blocked for {symbol}: {reason}",
            severity="CRITICAL",
            symbol=symbol,
        )
    
    def unblock_symbol(self, symbol: str) -> None:
        """Unblock a symbol."""
        with self._lock:
            self._blocked_symbols.discard(symbol)
        
        logger.info(
            f"Trading unblocked for {symbol}",
            category=LogCategory.RISK,
            symbol=symbol,
        )
    
    def add_limit(self, limit: PositionLimitConfig) -> None:
        """Add a custom limit."""
        with self._lock:
            self._limits.append(limit)
    
    def get_utilization(self) -> Dict[str, float]:
        """
        Get current limit utilization percentages.
        
        Useful for monitoring and dashboards.
        """
        with self._lock:
            gross = sum(abs(p) for p in self._positions.values())
            net = sum(p for p in self._positions.values())
            
            utilization = {}
            
            for limit in self._limits:
                if limit.limit_type == LimitType.GROSS:
                    utilization["gross"] = gross / limit.limit_value if limit.limit_value > 0 else 0
                elif limit.limit_type == LimitType.NET:
                    utilization["net"] = abs(net) / limit.limit_value if limit.limit_value > 0 else 0
            
            # Per-symbol utilization
            for symbol, pos in self._positions.items():
                symbol_limit = self._config.max_position_per_symbol
                utilization[f"symbol_{symbol}"] = abs(pos) / symbol_limit if symbol_limit > 0 else 0
            
            return utilization
    
    def get_breach_history(self, limit: int = 100) -> List[LimitBreach]:
        """Get recent breach history."""
        with self._lock:
            return list(self._breaches[-limit:])
    
    def reset(self) -> None:
        """Reset all positions and state."""
        with self._lock:
            self._positions.clear()
            self._breaches.clear()
            self._blocked_symbols.clear()
