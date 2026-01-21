"""
HFT Research System - Main Entry Point
=======================================

This is the main entry point for the HFT research system.
It provides several modes of operation:

1. BACKTEST MODE: Run historical simulation
2. LIVE SIMULATION MODE: Run with real-time simulated data
3. RESEARCH MODE: Run with live (delayed) Yahoo Finance data

Usage:
    python -m hft_system.main --mode backtest --symbols AAPL,MSFT,SPY
    python -m hft_system.main --mode simulation --duration 60
    python -m hft_system.main --mode research --symbols AAPL

For detailed options:
    python -m hft_system.main --help
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from typing import List, Optional

# Import system components
from .infra import (
    SystemConfig,
    get_default_config,
    get_backtest_config,
    logger,
    LogCategory,
    PerformanceTracker,
    format_metrics_report,
)
from .data import (
    create_data_feed,
    SimulatedDataFeed,
    MarketDataEvent,
    OrderBookSimulator,
)
from .infra.config import DataSource
from .signals import SignalAggregator, CombinedSignal
from .execution import SmartOrderRouter, ExecutionSimulator
from .risk import PositionLimitManager, KillSwitch, TradingState
from .backtest import EventDrivenBacktester, run_quick_backtest


# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n⚠️  Shutdown requested, cleaning up...")


async def run_backtest_mode(
    symbols: List[str],
    duration_minutes: int,
    seed: int
) -> None:
    """
    Run the system in backtest mode.
    
    This performs a historical simulation with synthetic data,
    testing the full trading pipeline.
    """
    print("\n" + "="*60)
    print("🚀 HFT RESEARCH SYSTEM - BACKTEST MODE")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    # Create backtester
    config = get_backtest_config()
    backtester = EventDrivenBacktester(config, seed=seed)
    
    # Run backtest
    result = await backtester.run_backtest(
        symbols=symbols,
        duration_minutes=duration_minutes
    )
    
    # Print results
    backtester.print_report(result)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BACKTEST SUMMARY")
    print("="*60)
    print(f"Total PnL:      ${result.total_pnl:,.2f}")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"Hit Rate:       {result.hit_rate:.1%}")
    print(f"Total Trades:   {result.num_trades:,}")
    print(f"Avg Slippage:   {result.avg_slippage_bps:.2f} bps")
    print("="*60)


async def run_simulation_mode(
    symbols: List[str],
    duration_seconds: int
) -> None:
    """
    Run the system in live simulation mode.
    
    Uses simulated data feed but runs in real-time
    to test system behavior under realistic conditions.
    """
    print("\n" + "="*60)
    print("🔴 HFT RESEARCH SYSTEM - LIVE SIMULATION MODE")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {duration_seconds} seconds")
    print("="*60 + "\n")
    
    # Initialize components
    config = get_default_config()
    config.market_data.source = DataSource.SIMULATED
    config.market_data.symbols = symbols
    
    performance_tracker = PerformanceTracker()
    signal_aggregator = SignalAggregator(config.signals)
    order_router = SmartOrderRouter(config.execution)
    execution_simulator = ExecutionSimulator(
        config.execution, config.latency, performance_tracker
    )
    position_manager = PositionLimitManager(config.risk, performance_tracker)
    kill_switch = KillSwitch(config.risk, performance_tracker)
    
    # Set up latency arbitrage pairs
    if "SPY" in symbols:
        for symbol in symbols:
            if symbol != "SPY":
                signal_aggregator.add_latency_pair("SPY", symbol)
    
    # Create data feed
    data_feed = SimulatedDataFeed(
        config.market_data,
        config.latency,
        seed=42
    )
    
    # Event counter
    event_count = 0
    trade_count = 0
    
    async def handle_market_data(event: MarketDataEvent):
        """Process incoming market data."""
        nonlocal event_count, trade_count
        event_count += 1
        
        if shutdown_requested or kill_switch.is_halted:
            return
        
        quote = event.data
        symbol = event.symbol
        
        # Update components
        execution_simulator.update_market_data(symbol, quote=quote)
        order_router.update_market_data(symbol, quote=quote)
        performance_tracker.update_price(symbol, quote.mid_price)
        kill_switch.update_data_timestamp(symbol)
        
        # Generate signal
        signal = signal_aggregator.process_quote(quote)
        
        # Trade if signal is strong
        if signal and signal.should_trade():
            # Check risk limits
            side = signal.suggested_side
            allowed, _, _ = position_manager.check_order(
                symbol, side, config.execution.default_order_size
            )
            
            if allowed:
                # Route order
                order = await order_router.route_signal(
                    signal, 
                    config.risk.max_position_per_symbol
                )
                
                if order:
                    # Execute
                    result = await execution_simulator.execute_order(
                        order, order_router
                    )
                    
                    if result.success:
                        trade_count += 1
                        signed_qty = result.filled_quantity
                        if order.side.value == "SELL":
                            signed_qty = -signed_qty
                        position_manager.update_position(symbol, signed_qty)
                        
                        # Update kill switch
                        metrics = performance_tracker.get_metrics()
                        kill_switch.update_pnl(metrics.get("total_pnl", 0.0))
        
        # Periodic status
        if event_count % 100 == 0:
            metrics = performance_tracker.get_metrics()
            print(f"\r📈 Events: {event_count:,} | Trades: {trade_count} | "
                  f"PnL: ${metrics.get('total_pnl', 0):,.2f}", end="")
    
    # Subscribe to data
    for symbol in symbols:
        data_feed.subscribe(symbol, handle_market_data)
    
    # Start feed
    await data_feed.start()
    
    print("🟢 Simulation started. Press Ctrl+C to stop.\n")
    
    # Run for specified duration
    try:
        await asyncio.sleep(duration_seconds)
    except asyncio.CancelledError:
        pass
    finally:
        await data_feed.stop()
    
    # Print final results
    print("\n\n")
    metrics = performance_tracker.get_metrics()
    print(format_metrics_report(metrics))


async def run_research_mode(
    symbols: List[str],
    duration_seconds: int
) -> None:
    """
    Run the system in research mode with Yahoo Finance data.
    
    NOTE: Yahoo Finance data is delayed and NOT suitable for
    actual trading decisions. This mode is for research and
    demonstration purposes only.
    """
    print("\n" + "="*60)
    print("🔬 HFT RESEARCH SYSTEM - RESEARCH MODE")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {duration_seconds} seconds")
    print("="*60)
    print("\n⚠️  WARNING: Using Yahoo Finance data (delayed quotes)")
    print("⚠️  This mode is for RESEARCH ONLY, not live trading\n")
    
    # Initialize components
    config = get_default_config()
    config.market_data.source = DataSource.YAHOO_FINANCE
    config.market_data.symbols = symbols
    config.market_data.poll_interval_seconds = 2.0  # Respect rate limits
    
    try:
        from .data import YahooFinanceDataFeed
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return
    
    performance_tracker = PerformanceTracker()
    signal_aggregator = SignalAggregator(config.signals)
    
    # Create Yahoo Finance feed
    data_feed = YahooFinanceDataFeed(config.market_data, config.latency)
    
    event_count = 0
    
    async def handle_market_data(event: MarketDataEvent):
        """Process incoming market data."""
        nonlocal event_count
        event_count += 1
        
        if shutdown_requested:
            return
        
        quote = event.data
        symbol = event.symbol
        
        # Update tracker
        performance_tracker.update_price(symbol, quote.mid_price)
        
        # Generate signal (for analysis only)
        signal = signal_aggregator.process_quote(quote)
        
        # Log interesting signals
        if signal and abs(signal.signal_value) > 0.3:
            print(f"📊 {symbol}: Signal={signal.signal_value:+.3f} "
                  f"(conf={signal.confidence:.2f}) "
                  f"| Mid=${quote.mid_price:.2f} "
                  f"| Spread={quote.spread_bps:.1f}bps")
    
    # Subscribe to data
    for symbol in symbols:
        data_feed.subscribe(symbol, handle_market_data)
    
    # Start feed
    await data_feed.start()
    
    print("🟢 Research mode started. Press Ctrl+C to stop.\n")
    
    # Run for specified duration
    try:
        await asyncio.sleep(duration_seconds)
    except asyncio.CancelledError:
        pass
    finally:
        await data_feed.stop()
    
    # Print signal statistics
    print("\n\n" + "="*60)
    print("📊 SIGNAL STATISTICS")
    print("="*60)
    for symbol in symbols:
        stats = signal_aggregator.get_signal_statistics(symbol)
        if stats:
            print(f"\n{symbol}:")
            print(f"  Signals generated: {stats.get('count', 0)}")
            print(f"  Mean signal:       {stats.get('mean_signal', 0):.4f}")
            print(f"  Signal std:        {stats.get('std_signal', 0):.4f}")
            print(f"  Avg confidence:    {stats.get('mean_confidence', 0):.2f}")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HFT Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hft_system.main --mode backtest --symbols AAPL,MSFT,SPY --duration 60
  python -m hft_system.main --mode simulation --duration 120
  python -m hft_system.main --mode research --symbols AAPL,GOOGL
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "simulation", "research"],
        default="backtest",
        help="Operating mode (default: backtest)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,GOOGL,SPY",
        help="Comma-separated list of symbols (default: AAPL,MSFT,GOOGL,SPY)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in minutes (backtest) or seconds (simulation/research)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run appropriate mode
    if args.mode == "backtest":
        asyncio.run(run_backtest_mode(symbols, args.duration, args.seed))
    elif args.mode == "simulation":
        asyncio.run(run_simulation_mode(symbols, args.duration))
    elif args.mode == "research":
        asyncio.run(run_research_mode(symbols, args.duration))


if __name__ == "__main__":
    main()
