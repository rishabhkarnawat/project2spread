"""
Event-Driven Backtester for HFT Research System
================================================

This backtester implements TRUE event-driven simulation,
not the naive bar-based approach used in many systems.

WHY EVENT-DRIVEN BACKTESTING:
=============================

BAR-BASED (Bad):
- Groups data into time bars (1-min, 5-min)
- Assumes trades happen at bar close
- Ignores intra-bar price movement
- Can't model microstructure effects

EVENT-DRIVEN (Good):
- Processes each tick/quote/trade as it happens
- Realistic fill simulation
- Captures latency effects
- Models queue position and adverse selection

WHAT WE SIMULATE:
================

1. MARKET DATA EVENTS:
   - Quote updates (bid/ask changes)
   - Trade prints (executions in market)
   - Order book changes

2. TRADING EVENTS:
   - Signal generation
   - Order submission
   - Fill/cancel events
   - Position updates

3. TIMING:
   - Deterministic replay of historical events
   - Configurable latency simulation
   - Time acceleration for fast backtests

4. STRESS TESTING:
   - Latency spike injection
   - Volatility shocks
   - Data dropout simulation
   - Gap events

IMPORTANT CAVEATS:
=================
- Backtests are ALWAYS optimistic
- Market impact is estimated, not real
- Competition is not modeled (alpha decay)
- Black swan events are hard to simulate
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable, Any, Deque
from collections import deque
import heapq

import numpy as np

from ..data.market_data_feed import (
    MarketDataEvent, MarketDataType, Quote, SimulatedDataFeed
)
from ..data.orderbook_simulator import OrderBook, OrderBookSimulator
from ..signals.signal_aggregator import SignalAggregator, CombinedSignal
from ..execution.smart_order_router import SmartOrderRouter, Order, OrderSideEnum
from ..execution.execution_simulator import ExecutionSimulator, ExecutionResult
from ..risk.position_limits import PositionLimitManager
from ..risk.kill_switch import KillSwitch
from ..infra.config import SystemConfig, BacktestConfig, get_backtest_config
from ..infra.logging import get_logger, LogCategory
from ..infra.performance_metrics import PerformanceTracker, format_metrics_report


logger = get_logger()


class EventType(Enum):
    """Types of events in the simulation."""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    RISK_CHECK = "risk_check"
    HEARTBEAT = "heartbeat"


@dataclass(order=True)
class SimulationEvent:
    """
    An event in the simulation timeline.
    
    Events are processed in timestamp order to ensure
    deterministic replay.
    """
    timestamp: datetime
    event_type: EventType = field(compare=False)
    data: Any = field(compare=False)
    priority: int = field(default=0, compare=True)  # Lower = higher priority


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Performance metrics
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    
    # Trade statistics
    num_trades: int
    num_signals: int
    avg_trade_pnl: float
    avg_slippage_bps: float
    
    # Detailed metrics
    full_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Event counts
    events_processed: int = 0
    
    # PnL series for analysis
    pnl_series: List[tuple] = field(default_factory=list)


class EventDrivenBacktester:
    """
    Event-driven backtesting engine.
    
    Processes market events in timestamp order,
    generating signals, executing trades, and tracking performance.
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        seed: int = 42
    ):
        """
        Initialize backtester.
        
        Args:
            config: System configuration (uses backtest defaults if None)
            seed: Random seed for reproducibility
        """
        self._config = config or get_backtest_config()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        
        # Components
        self._performance_tracker = PerformanceTracker()
        self._signal_aggregator = SignalAggregator(self._config.signals)
        self._order_router = SmartOrderRouter(self._config.execution)
        self._execution_simulator = ExecutionSimulator(
            self._config.execution,
            self._config.latency,
            self._performance_tracker
        )
        self._position_manager = PositionLimitManager(
            self._config.risk,
            self._performance_tracker
        )
        self._kill_switch = KillSwitch(
            self._config.risk,
            self._performance_tracker
        )
        
        # Order book simulator
        self._book_simulator = OrderBookSimulator(
            self._config.market_data.symbols,
            seed=seed
        )
        
        # Event queue (min-heap by timestamp)
        self._event_queue: List[SimulationEvent] = []
        
        # Current simulation time
        self._current_time: datetime = datetime.utcnow()
        
        # Statistics
        self._events_processed: int = 0
        self._signals_generated: int = 0
        self._orders_placed: int = 0
        
        # Stress test parameters
        self._latency_multiplier: float = 1.0
        self._volatility_multiplier: float = 1.0
        self._dropout_active: bool = False
    
    def schedule_event(self, event: SimulationEvent) -> None:
        """Add event to the queue."""
        heapq.heappush(self._event_queue, event)
    
    def _get_next_event(self) -> Optional[SimulationEvent]:
        """Get next event from queue."""
        if self._event_queue:
            return heapq.heappop(self._event_queue)
        return None
    
    async def run_backtest(
        self,
        symbols: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_minutes: int = 60
    ) -> BacktestResult:
        """
        Run a full backtest simulation.
        
        Args:
            symbols: Symbols to trade
            start_time: Start of simulation
            end_time: End of simulation
            duration_minutes: Duration if end_time not specified
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(
            f"Starting backtest for {len(symbols)} symbols over {duration_minutes} minutes",
            category=LogCategory.SYSTEM
        )
        
        # Reset state
        self._reset()
        
        # Set time bounds
        if start_time is None:
            start_time = datetime.utcnow()
        if end_time is None:
            end_time = start_time + timedelta(minutes=duration_minutes)
        
        self._current_time = start_time
        
        # Initialize order books
        for symbol in symbols:
            # Use realistic starting prices
            initial_prices = {
                "AAPL": 185.0, "MSFT": 375.0, "GOOGL": 140.0, "SPY": 450.0,
                "AMZN": 150.0, "META": 350.0, "NVDA": 500.0, "TSLA": 250.0
            }
            mid_price = initial_prices.get(symbol, 100.0)
            self._book_simulator.initialize_book(symbol, mid_price)
            
            # Set volume estimates
            self._execution_simulator.set_volume_estimate(symbol, 10_000_000)
            self._execution_simulator.set_volatility(symbol, 0.02)
        
        # Set up latency arbitrage pairs
        if "SPY" in symbols:
            for symbol in symbols:
                if symbol != "SPY":
                    self._signal_aggregator.add_latency_pair(
                        "SPY", symbol, correlation=0.8, typical_lag_ms=20.0
                    )
        
        # Generate initial market data events
        self._generate_market_data_events(symbols, start_time, end_time)
        
        # Schedule periodic risk checks
        self._schedule_risk_checks(start_time, end_time)
        
        # Main event loop
        wall_start = time.time()
        
        while self._event_queue:
            event = self._get_next_event()
            if event is None:
                break
            
            # Check if past end time
            if event.timestamp > end_time:
                break
            
            # Update simulation time
            self._current_time = event.timestamp
            
            # Process event
            await self._process_event(event)
            
            self._events_processed += 1
            
            # Periodic progress logging
            if self._events_processed % 10000 == 0:
                elapsed = (self._current_time - start_time).total_seconds()
                total = (end_time - start_time).total_seconds()
                progress = elapsed / total * 100 if total > 0 else 0
                logger.debug(
                    f"Backtest progress: {progress:.1f}% ({self._events_processed} events)",
                    category=LogCategory.SYSTEM
                )
        
        wall_end = time.time()
        
        # Get final metrics
        metrics = self._performance_tracker.get_metrics()
        
        result = BacktestResult(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=wall_end - wall_start,
            total_pnl=metrics.get("total_pnl", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            hit_rate=metrics.get("hit_rate", 0.0),
            num_trades=int(metrics.get("num_trades", 0)),
            num_signals=self._signals_generated,
            avg_trade_pnl=metrics.get("avg_trade_pnl", 0.0),
            avg_slippage_bps=metrics.get("avg_slippage_bps", 0.0),
            full_metrics=metrics,
            events_processed=self._events_processed,
            pnl_series=self._performance_tracker.get_pnl_series(),
        )
        
        # Log summary
        logger.info(
            f"Backtest complete: {result.num_trades} trades, "
            f"PnL=${result.total_pnl:.2f}, Sharpe={result.sharpe_ratio:.2f}",
            category=LogCategory.SYSTEM
        )
        
        return result
    
    def _generate_market_data_events(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Generate simulated market data events.
        
        Creates a stream of quote updates for all symbols
        throughout the simulation period.
        """
        tick_interval_ms = 1000 / self._config.market_data.simulated_ticks_per_second
        
        current = start_time
        sequence = 0
        
        while current < end_time:
            for symbol in symbols:
                # Skip if dropout is active
                if self._dropout_active and self._rng.random() < 0.1:
                    continue
                
                # Generate quote
                book = self._book_simulator.get_book(symbol)
                if book:
                    quote = self._generate_quote_from_book(book, current)
                else:
                    # Initialize if needed
                    book = self._book_simulator.initialize_book(symbol, 100.0)
                    quote = self._generate_quote_from_book(book, current)
                
                # Create market data event
                event = SimulationEvent(
                    timestamp=current,
                    event_type=EventType.MARKET_DATA,
                    data=MarketDataEvent(
                        event_type=MarketDataType.QUOTE,
                        symbol=symbol,
                        timestamp=current,
                        data=quote,
                        sequence_number=sequence,
                    ),
                    priority=1,  # Market data is high priority
                )
                
                self.schedule_event(event)
                sequence += 1
            
            # Advance time
            current += timedelta(milliseconds=tick_interval_ms)
            
            # Evolve order books
            self._evolve_order_books(symbols)
    
    def _generate_quote_from_book(
        self,
        book: OrderBook,
        timestamp: datetime
    ) -> Quote:
        """Generate a quote from order book state."""
        return Quote(
            symbol=book.symbol,
            bid_price=book.best_bid_price,
            bid_size=book.best_bid.size if book.best_bid else 0,
            ask_price=book.best_ask_price,
            ask_size=book.best_ask.size if book.best_ask else 0,
            timestamp=timestamp,
        )
    
    def _evolve_order_books(self, symbols: List[str]) -> None:
        """Evolve order books based on simulated dynamics."""
        for symbol in symbols:
            book = self._book_simulator.get_book(symbol)
            if not book:
                continue
            
            # Simulate price movement
            mid = book.mid_price
            volatility = 0.02 * self._volatility_multiplier
            tick_vol = volatility / np.sqrt(252 * 6.5 * 3600 * self._config.market_data.simulated_ticks_per_second)
            
            # Random return
            ret = self._rng.normal(0, tick_vol)
            new_mid = mid * np.exp(ret)
            
            # Update spread (widens with volatility)
            base_spread_bps = 5.0 * (1 + abs(ret) * 100)
            spread = new_mid * base_spread_bps / 10000
            
            # Generate new quote
            new_bid = new_mid - spread / 2
            new_ask = new_mid + spread / 2
            
            # Random sizes
            bid_size = max(100, int(500 + self._rng.normal(0, 100)))
            ask_size = max(100, int(500 + self._rng.normal(0, 100)))
            
            # Update book
            self._book_simulator.update_from_quote(
                symbol,
                round(new_bid, 4),
                bid_size,
                round(new_ask, 4),
                ask_size
            )
    
    def _schedule_risk_checks(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """Schedule periodic risk checks throughout simulation."""
        current = start_time
        interval = timedelta(seconds=1)  # Risk check every second
        
        while current < end_time:
            event = SimulationEvent(
                timestamp=current,
                event_type=EventType.RISK_CHECK,
                data=None,
                priority=0,  # Highest priority
            )
            self.schedule_event(event)
            current += interval
    
    async def _process_event(self, event: SimulationEvent) -> None:
        """
        Process a single event in the simulation.
        
        This is the core of the event-driven architecture.
        Each event can generate new events (e.g., market data -> signal -> order).
        """
        try:
            if event.event_type == EventType.MARKET_DATA:
                await self._handle_market_data(event.data)
                
            elif event.event_type == EventType.SIGNAL:
                await self._handle_signal(event.data)
                
            elif event.event_type == EventType.ORDER:
                await self._handle_order(event.data)
                
            elif event.event_type == EventType.FILL:
                await self._handle_fill(event.data)
                
            elif event.event_type == EventType.RISK_CHECK:
                await self._handle_risk_check()
                
            elif event.event_type == EventType.HEARTBEAT:
                self._kill_switch.heartbeat()
                
        except Exception as e:
            logger.error(
                f"Error processing event {event.event_type}: {e}",
                category=LogCategory.SYSTEM
            )
            self._kill_switch.record_error()
    
    async def _handle_market_data(self, md_event: MarketDataEvent) -> None:
        """Handle market data event."""
        symbol = md_event.symbol
        quote = md_event.data
        
        # Update execution simulator
        self._execution_simulator.update_market_data(symbol, quote=quote)
        self._order_router.update_market_data(symbol, quote=quote)
        
        # Update performance tracker with current price
        self._performance_tracker.update_price(symbol, quote.mid_price)
        
        # Update kill switch
        self._kill_switch.update_data_timestamp(symbol)
        
        # Generate signal
        signal = self._signal_aggregator.process_quote(quote)
        
        if signal and signal.should_trade(self._config.signals.min_confidence_to_trade):
            self._signals_generated += 1
            
            # Schedule signal processing with latency
            latency_ms = self._config.latency.signal_compute_latency_ms * self._latency_multiplier
            signal_time = self._current_time + timedelta(milliseconds=latency_ms)
            
            self.schedule_event(SimulationEvent(
                timestamp=signal_time,
                event_type=EventType.SIGNAL,
                data=(symbol, signal),
                priority=2,
            ))
    
    async def _handle_signal(self, signal_data: tuple) -> None:
        """Handle a trading signal."""
        symbol, signal = signal_data
        
        # Check kill switch
        if not self._kill_switch.is_active:
            return
        
        # Check position limits
        side = "BUY" if signal.suggested_side == "BUY" else "SELL"
        max_position = self._config.risk.max_position_per_symbol
        
        allowed, adjusted_qty, breaches = self._position_manager.check_order(
            symbol, side, int(max_position * signal.suggested_size_pct)
        )
        
        if not allowed:
            return
        
        # Route signal to order
        order = await self._order_router.route_signal(signal, max_position)
        
        if order:
            self._orders_placed += 1
            
            # Schedule order execution with latency
            latency_ms = self._config.latency.base_latency_ms * self._latency_multiplier
            order_time = self._current_time + timedelta(milliseconds=latency_ms)
            
            self.schedule_event(SimulationEvent(
                timestamp=order_time,
                event_type=EventType.ORDER,
                data=order,
                priority=3,
            ))
    
    async def _handle_order(self, order: Order) -> None:
        """Handle order submission."""
        # Execute order through simulator
        result = await self._execution_simulator.execute_order(
            order, self._order_router
        )
        
        if result.success:
            # Update position
            signed_qty = result.filled_quantity
            if order.side == OrderSideEnum.SELL:
                signed_qty = -signed_qty
            
            self._position_manager.update_position(order.symbol, signed_qty)
            
            # Update kill switch with PnL
            metrics = self._performance_tracker.get_metrics()
            self._kill_switch.update_pnl(metrics.get("total_pnl", 0.0))
            
            # Update latency tracking
            self._kill_switch.update_latency(result.latency_ms)
    
    async def _handle_fill(self, fill_data: dict) -> None:
        """Handle fill notification."""
        # Already handled in _handle_order for simplicity
        pass
    
    async def _handle_risk_check(self) -> None:
        """Handle periodic risk check."""
        # Run kill switch checks
        await self._kill_switch.check_all()
        
        # Send heartbeat
        self._kill_switch.heartbeat()
    
    def inject_latency_spike(self, multiplier: float = 5.0, duration_ms: float = 1000) -> None:
        """
        Inject a latency spike for stress testing.
        
        Simulates network congestion or system overload.
        """
        self._latency_multiplier = multiplier
        logger.warning(
            f"Injected latency spike: {multiplier}x",
            category=LogCategory.SYSTEM
        )
        
        # Schedule recovery
        recovery_time = self._current_time + timedelta(milliseconds=duration_ms)
        self.schedule_event(SimulationEvent(
            timestamp=recovery_time,
            event_type=EventType.HEARTBEAT,  # Reuse for recovery
            data={"action": "recover_latency"},
            priority=0,
        ))
    
    def inject_volatility_shock(self, multiplier: float = 3.0) -> None:
        """
        Inject a volatility shock for stress testing.
        
        Simulates flash crash or news event.
        """
        self._volatility_multiplier = multiplier
        logger.warning(
            f"Injected volatility shock: {multiplier}x",
            category=LogCategory.SYSTEM
        )
    
    def inject_data_dropout(self, duration_ms: float = 500) -> None:
        """
        Inject a data feed dropout for stress testing.
        
        Simulates feed handler failure or network issue.
        """
        self._dropout_active = True
        logger.warning(
            f"Injected data dropout for {duration_ms}ms",
            category=LogCategory.SYSTEM
        )
        
        # Schedule recovery
        recovery_time = self._current_time + timedelta(milliseconds=duration_ms)
        self.schedule_event(SimulationEvent(
            timestamp=recovery_time,
            event_type=EventType.HEARTBEAT,
            data={"action": "recover_dropout"},
            priority=0,
        ))
    
    def _reset(self) -> None:
        """Reset backtester state for new run."""
        self._event_queue.clear()
        self._events_processed = 0
        self._signals_generated = 0
        self._orders_placed = 0
        self._latency_multiplier = 1.0
        self._volatility_multiplier = 1.0
        self._dropout_active = False
        
        self._performance_tracker.reset()
        self._signal_aggregator.reset()
        self._position_manager.reset()
        self._kill_switch.reset()
        
        self._rng = np.random.default_rng(self._seed)
    
    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report."""
        print(format_metrics_report(result.full_metrics))
        
        print(f"\nBacktest Duration: {result.duration_seconds:.2f} seconds")
        print(f"Events Processed: {result.events_processed:,}")
        print(f"Signals Generated: {result.num_signals:,}")
        print(f"Events/Second: {result.events_processed / result.duration_seconds:,.0f}")


async def run_quick_backtest(
    symbols: List[str] = None,
    duration_minutes: int = 10
) -> BacktestResult:
    """
    Convenience function to run a quick backtest.
    
    Args:
        symbols: Symbols to trade (defaults to common ones)
        duration_minutes: How long to simulate
        
    Returns:
        BacktestResult
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]
    
    backtester = EventDrivenBacktester()
    result = await backtester.run_backtest(
        symbols=symbols,
        duration_minutes=duration_minutes
    )
    
    backtester.print_report(result)
    return result
