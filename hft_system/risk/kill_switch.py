"""
Kill Switch for HFT Research System
====================================

The kill switch is the LAST LINE OF DEFENSE against catastrophic losses.
It monitors system health and trading performance, halting trading
when anomalies are detected.

WHY KILL SWITCHES MATTER:
========================

KNIGHT CAPITAL (2012):
- Software bug deployed to production
- Lost $440 MILLION in 45 minutes
- Adequate kill switch could have limited losses to ~$10M

FLASH CRASH (2010):
- Dow dropped 1000 points in minutes
- Many firms hit kill switches and recovered
- Those without suffered larger losses

KILL SWITCH TRIGGERS:
====================

1. LOSS LIMITS:
   - Max loss per second
   - Max loss per minute
   - Max daily loss
   - Max loss per trade

2. POSITION TRIGGERS:
   - Position exceeds threshold
   - Gross exposure too high
   - Inventory accumulation

3. MARKET CONDITIONS:
   - Extreme volatility
   - Wide spreads
   - Data feed issues
   - Unusual price moves

4. SYSTEM HEALTH:
   - High latency
   - Error rate
   - Fill rate anomalies
   - Network issues

KILL SWITCH BEHAVIOR:
====================
When triggered, the kill switch should:
1. CANCEL all pending orders immediately
2. OPTIONALLY flatten positions
3. PREVENT new orders
4. ALERT humans
5. LOG all state for post-mortem

This is NOT negotiable - when kill switch triggers, STOP TRADING.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable, Deque
import threading

from ..infra.config import RiskConfig
from ..infra.logging import get_logger, LogCategory
from ..infra.performance_metrics import PerformanceTracker


logger = get_logger()


class KillSwitchReason(Enum):
    """Reasons for kill switch activation."""
    LOSS_LIMIT = "loss_limit"
    POSITION_LIMIT = "position_limit"
    VOLATILITY = "volatility"
    DATA_FEED = "data_feed"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    HEARTBEAT = "heartbeat"


@dataclass
class KillSwitchEvent:
    """Record of a kill switch activation."""
    timestamp: datetime
    reason: KillSwitchReason
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    auto_recovered: bool = False


class TradingState(Enum):
    """Current trading state."""
    ACTIVE = "active"           # Normal trading
    HALTED = "halted"           # Kill switch triggered
    CAUTIOUS = "cautious"       # Reduced trading (warnings)
    RECOVERING = "recovering"   # After halt, before full active


# Callback for kill switch events
KillSwitchCallback = Callable[[KillSwitchEvent], Awaitable[None]]


class KillSwitch:
    """
    Kill switch that monitors system health and halts trading on anomalies.
    
    This is a critical safety component that must:
    - Be independent of trading logic
    - Have fail-safe defaults (halt if uncertain)
    - Be impossible to accidentally disable
    - Log everything for post-mortem analysis
    """
    
    def __init__(
        self,
        config: RiskConfig,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize kill switch.
        
        Args:
            config: Risk configuration with thresholds
            performance_tracker: For monitoring PnL
        """
        self._config = config
        self._tracker = performance_tracker
        
        # Current state
        self._state = TradingState.ACTIVE
        self._state_lock = threading.Lock()
        
        # Event history
        self._events: List[KillSwitchEvent] = []
        
        # PnL tracking
        self._pnl_history: Deque[tuple] = deque(maxlen=10000)  # (timestamp, pnl)
        self._last_pnl: float = 0.0
        
        # Loss tracking windows
        self._second_losses: Deque[tuple] = deque(maxlen=100)
        self._minute_losses: Deque[tuple] = deque(maxlen=100)
        self._daily_loss: float = 0.0
        self._day_start: datetime = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
        # Volatility tracking
        self._recent_returns: Deque[float] = deque(maxlen=1000)
        
        # Data feed monitoring
        self._last_data_times: Dict[str, datetime] = {}
        self._data_timeout_seconds: float = 5.0
        
        # Latency monitoring
        self._latency_samples: Deque[float] = deque(maxlen=100)
        self._max_acceptable_latency_ms: float = 100.0
        
        # Error tracking
        self._error_count: int = 0
        self._error_window_start: datetime = datetime.utcnow()
        self._max_errors_per_minute: int = 10
        
        # Heartbeat
        self._last_heartbeat: datetime = datetime.utcnow()
        self._heartbeat_timeout_seconds: float = 30.0
        
        # Callbacks
        self._callbacks: List[KillSwitchCallback] = []
        
        # Recovery settings
        self._recovery_cooldown_seconds: float = 60.0
        self._last_halt_time: Optional[datetime] = None
    
    @property
    def state(self) -> TradingState:
        """Get current trading state."""
        with self._state_lock:
            return self._state
    
    @property
    def is_active(self) -> bool:
        """Check if trading is allowed."""
        return self.state == TradingState.ACTIVE
    
    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self.state == TradingState.HALTED
    
    def register_callback(self, callback: KillSwitchCallback) -> None:
        """Register callback for kill switch events."""
        self._callbacks.append(callback)
    
    async def check_all(self) -> bool:
        """
        Run all kill switch checks.
        
        Should be called regularly (e.g., every trade or every second).
        
        Returns:
            True if trading is allowed, False if halted
        """
        if self.is_halted:
            return False
        
        # Check each trigger
        checks = [
            self._check_loss_limits(),
            self._check_position_limits(),
            self._check_volatility(),
            self._check_data_feed(),
            self._check_latency(),
            self._check_errors(),
            self._check_heartbeat(),
        ]
        
        for check_result in checks:
            if check_result:
                return False  # Halt triggered
        
        return True
    
    def update_pnl(self, pnl: float) -> bool:
        """
        Update PnL and check loss limits.
        
        Called after each trade or on mark-to-market.
        
        Returns:
            True if OK, False if kill switch triggered
        """
        now = datetime.utcnow()
        
        # Calculate change
        pnl_change = pnl - self._last_pnl
        self._last_pnl = pnl
        
        # Track in history
        self._pnl_history.append((now, pnl))
        
        # Track losses (only negative changes)
        if pnl_change < 0:
            loss = abs(pnl_change)
            self._second_losses.append((now, loss))
            self._minute_losses.append((now, loss))
            
            # Update daily loss
            if now.date() > self._day_start.date():
                self._daily_loss = 0.0
                self._day_start = now.replace(hour=0, minute=0, second=0)
            self._daily_loss += loss
        
        return self._check_loss_limits() is None
    
    def _check_loss_limits(self) -> Optional[KillSwitchEvent]:
        """Check if loss limits are breached."""
        now = datetime.utcnow()
        
        # Loss per second
        one_second_ago = now - timedelta(seconds=1)
        second_loss = sum(
            loss for ts, loss in self._second_losses 
            if ts > one_second_ago
        )
        
        if second_loss > self._config.max_loss_per_second:
            return self._trigger(
                KillSwitchReason.LOSS_LIMIT,
                f"Loss per second ({second_loss:.2f}) exceeds limit ({self._config.max_loss_per_second:.2f})",
                {"loss_per_second": second_loss}
            )
        
        # Loss per minute
        one_minute_ago = now - timedelta(minutes=1)
        minute_loss = sum(
            loss for ts, loss in self._minute_losses 
            if ts > one_minute_ago
        )
        
        if minute_loss > self._config.max_loss_per_minute:
            return self._trigger(
                KillSwitchReason.LOSS_LIMIT,
                f"Loss per minute ({minute_loss:.2f}) exceeds limit ({self._config.max_loss_per_minute:.2f})",
                {"loss_per_minute": minute_loss}
            )
        
        # Daily loss
        if self._daily_loss > self._config.max_daily_loss:
            return self._trigger(
                KillSwitchReason.LOSS_LIMIT,
                f"Daily loss ({self._daily_loss:.2f}) exceeds limit ({self._config.max_daily_loss:.2f})",
                {"daily_loss": self._daily_loss}
            )
        
        return None
    
    def _check_position_limits(self) -> Optional[KillSwitchEvent]:
        """Check if position limits are critically breached."""
        if not self._tracker:
            return None
        
        gross = self._tracker.get_gross_position()
        
        if gross > self._config.kill_switch_position_threshold:
            return self._trigger(
                KillSwitchReason.POSITION_LIMIT,
                f"Gross position ({gross}) exceeds kill switch threshold ({self._config.kill_switch_position_threshold})",
                {"gross_position": gross}
            )
        
        return None
    
    def _check_volatility(self) -> Optional[KillSwitchEvent]:
        """Check for extreme volatility."""
        if len(self._recent_returns) < 10:
            return None
        
        # Calculate realized volatility
        import numpy as np
        returns = list(self._recent_returns)
        vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        if vol > self._config.extreme_volatility_threshold * 10:  # 10x threshold is extreme
            return self._trigger(
                KillSwitchReason.VOLATILITY,
                f"Extreme volatility detected ({vol:.2%})",
                {"volatility": vol}
            )
        
        return None
    
    def update_volatility(self, return_value: float) -> None:
        """Record a return for volatility monitoring."""
        self._recent_returns.append(return_value)
    
    def _check_data_feed(self) -> Optional[KillSwitchEvent]:
        """Check for data feed issues."""
        now = datetime.utcnow()
        
        for symbol, last_time in self._last_data_times.items():
            age = (now - last_time).total_seconds()
            if age > self._data_timeout_seconds:
                return self._trigger(
                    KillSwitchReason.DATA_FEED,
                    f"Data feed stale for {symbol} ({age:.1f}s)",
                    {"symbol": symbol, "stale_seconds": age}
                )
        
        return None
    
    def update_data_timestamp(self, symbol: str) -> None:
        """Record that data was received for a symbol."""
        self._last_data_times[symbol] = datetime.utcnow()
    
    def _check_latency(self) -> Optional[KillSwitchEvent]:
        """Check for latency issues."""
        if not self._latency_samples:
            return None
        
        import numpy as np
        avg_latency = np.mean(list(self._latency_samples))
        
        if avg_latency > self._max_acceptable_latency_ms:
            return self._trigger(
                KillSwitchReason.LATENCY,
                f"High latency detected ({avg_latency:.1f}ms)",
                {"avg_latency_ms": avg_latency}
            )
        
        return None
    
    def update_latency(self, latency_ms: float) -> None:
        """Record latency measurement."""
        self._latency_samples.append(latency_ms)
    
    def _check_errors(self) -> Optional[KillSwitchEvent]:
        """Check error rate."""
        now = datetime.utcnow()
        
        # Reset window if needed
        if (now - self._error_window_start).total_seconds() > 60:
            self._error_count = 0
            self._error_window_start = now
        
        if self._error_count > self._max_errors_per_minute:
            return self._trigger(
                KillSwitchReason.ERROR_RATE,
                f"High error rate ({self._error_count} errors in last minute)",
                {"error_count": self._error_count}
            )
        
        return None
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self._error_count += 1
    
    def _check_heartbeat(self) -> Optional[KillSwitchEvent]:
        """Check heartbeat timeout."""
        now = datetime.utcnow()
        age = (now - self._last_heartbeat).total_seconds()
        
        if age > self._heartbeat_timeout_seconds:
            return self._trigger(
                KillSwitchReason.HEARTBEAT,
                f"Heartbeat timeout ({age:.1f}s since last heartbeat)",
                {"seconds_since_heartbeat": age}
            )
        
        return None
    
    def heartbeat(self) -> None:
        """Send heartbeat to prevent timeout."""
        self._last_heartbeat = datetime.utcnow()
    
    def _trigger(
        self,
        reason: KillSwitchReason,
        message: str,
        metrics: Dict[str, float]
    ) -> KillSwitchEvent:
        """
        Trigger the kill switch.
        
        This is the actual halt action.
        """
        event = KillSwitchEvent(
            timestamp=datetime.utcnow(),
            reason=reason,
            message=message,
            metrics=metrics,
        )
        
        with self._state_lock:
            self._state = TradingState.HALTED
            self._last_halt_time = datetime.utcnow()
        
        self._events.append(event)
        
        # Log critically
        logger.critical(
            f"🚨 KILL SWITCH TRIGGERED: {reason.value}",
            category=LogCategory.RISK,
            reason=reason.value,
            message=message,
            **metrics
        )
        
        # Fire callbacks asynchronously
        asyncio.create_task(self._fire_callbacks(event))
        
        return event
    
    async def _fire_callbacks(self, event: KillSwitchEvent) -> None:
        """Fire all registered callbacks."""
        for callback in self._callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
    
    def manual_halt(self, reason: str = "Manual halt") -> KillSwitchEvent:
        """
        Manually trigger the kill switch.
        
        Used by operators or automated systems.
        """
        return self._trigger(
            KillSwitchReason.MANUAL,
            reason,
            {}
        )
    
    async def recover(self, force: bool = False) -> bool:
        """
        Attempt to recover from halt state.
        
        Recovery requires:
        1. Cooldown period elapsed
        2. All checks passing
        3. Explicit approval (force=True) or automatic recovery enabled
        
        Returns:
            True if recovered, False if still halted
        """
        if not self.is_halted:
            return True
        
        now = datetime.utcnow()
        
        # Check cooldown
        if self._last_halt_time:
            elapsed = (now - self._last_halt_time).total_seconds()
            if elapsed < self._recovery_cooldown_seconds and not force:
                logger.warning(
                    f"Recovery cooldown: {self._recovery_cooldown_seconds - elapsed:.0f}s remaining",
                    category=LogCategory.RISK
                )
                return False
        
        # Run all checks
        if await self.check_all():
            with self._state_lock:
                self._state = TradingState.ACTIVE
            
            logger.info(
                "✅ Kill switch recovered, trading resumed",
                category=LogCategory.RISK
            )
            
            # Update last event
            if self._events:
                self._events[-1].auto_recovered = True
            
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            "state": self.state.value,
            "is_active": self.is_active,
            "is_halted": self.is_halted,
            "total_events": len(self._events),
            "last_event": self._events[-1].__dict__ if self._events else None,
            "daily_loss": self._daily_loss,
            "error_count": self._error_count,
        }
    
    def get_events(self, limit: int = 100) -> List[KillSwitchEvent]:
        """Get recent kill switch events."""
        return list(self._events[-limit:])
    
    def reset(self) -> None:
        """
        Reset kill switch state.
        
        USE WITH CAUTION - only for backtesting or development.
        """
        with self._state_lock:
            self._state = TradingState.ACTIVE
        
        self._events.clear()
        self._pnl_history.clear()
        self._second_losses.clear()
        self._minute_losses.clear()
        self._daily_loss = 0.0
        self._last_pnl = 0.0
        self._error_count = 0
        self._recent_returns.clear()
        self._latency_samples.clear()
        self._last_data_times.clear()
        self._last_heartbeat = datetime.utcnow()
        self._last_halt_time = None
        
        logger.warning(
            "Kill switch state RESET",
            category=LogCategory.RISK
        )
