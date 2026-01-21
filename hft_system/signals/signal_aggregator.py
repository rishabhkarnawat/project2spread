"""
Signal Aggregator for HFT Research System
==========================================

This module combines multiple alpha signals into a unified
trading signal with proper weighting and confidence scoring.

SIGNAL COMBINATION THEORY:
==========================

Multiple signals should be combined because:
1. Diversification: Different signals capture different effects
2. Robustness: Single signal failure doesn't break system
3. Regime adaptation: Different signals work in different conditions

COMBINATION METHODS:
1. Simple weighted average (what we use)
2. Machine learning (ensemble models)
3. Dynamic weighting (regime-switching)
4. Risk-parity weighting (equal risk contribution)

OUR APPROACH:
- Weighted average of signal values
- Confidence-weighted (higher confidence = more weight)
- Normalized to [-1, +1] range
- Combined confidence from all signals

PRODUCTION CONSIDERATIONS:
- Signal combination should be < 100 nanoseconds
- Pre-computed weights
- SIMD vectorization
- No memory allocation in hot path
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque, Callable, Any

from .microprice import MicropriceSignal, MicropriceCalculator
from .order_flow_imbalance import OFISignal, OrderFlowImbalanceCalculator
from .latency_arbitrage import LatencyArbSignal, LatencyArbitrageDetector
from ..data.orderbook_simulator import OrderBook
from ..data.market_data_feed import Quote, MarketDataEvent, MarketDataType
from ..infra.logging import get_logger, LogCategory
from ..infra.config import SignalConfig


logger = get_logger()


@dataclass
class CombinedSignal:
    """
    The final combined trading signal.
    
    This is what gets passed to the execution engine
    for trading decisions.
    """
    symbol: str
    timestamp: datetime
    
    # Combined signal
    signal_value: float           # -1 to +1, final trading direction
    confidence: float             # Combined confidence
    
    # Component signals (for analysis)
    microprice_signal: Optional[float] = None
    ofi_signal: Optional[float] = None
    latency_arb_signal: Optional[float] = None
    
    # Component confidences
    microprice_confidence: float = 0.0
    ofi_confidence: float = 0.0
    latency_arb_confidence: float = 0.0
    
    # Execution hints
    urgency: str = "NORMAL"       # "LOW", "NORMAL", "HIGH"
    suggested_side: str = "NONE"  # "BUY", "SELL", "NONE"
    suggested_size_pct: float = 0.0  # Percentage of max position
    
    # Market conditions
    spread_bps: float = 0.0
    volatility: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_value": self.signal_value,
            "confidence": self.confidence,
            "microprice": self.microprice_signal,
            "ofi": self.ofi_signal,
            "latency_arb": self.latency_arb_signal,
            "urgency": self.urgency,
            "suggested_side": self.suggested_side,
        }
    
    def should_trade(self, min_confidence: float = 0.6) -> bool:
        """Determine if signal is strong enough to trade."""
        return (
            abs(self.signal_value) > 0.2 and 
            self.confidence >= min_confidence
        )


class SignalAggregator:
    """
    Aggregates multiple signals into a unified trading signal.
    
    This is the central coordinator for signal generation,
    responsible for:
    1. Routing market data to appropriate signal calculators
    2. Combining signals with configurable weights
    3. Generating trading decisions
    """
    
    def __init__(self, config: SignalConfig):
        """
        Initialize signal aggregator.
        
        Args:
            config: Signal configuration with weights and thresholds
        """
        self._config = config
        
        # Initialize signal calculators
        self._microprice_calc = MicropriceCalculator(config)
        self._ofi_calc = OrderFlowImbalanceCalculator(config)
        self._latency_arb = LatencyArbitrageDetector(config)
        
        # Signal weights (should sum to 1)
        self._weights = config.signal_weights.copy()
        self._normalize_weights()
        
        # Signal history for analysis
        self._signal_history: Dict[str, Deque[CombinedSignal]] = {}
        
        # Thresholds
        self._min_confidence = config.min_confidence_to_trade
        self._high_confidence = config.high_confidence_threshold
    
    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1."""
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}
    
    def add_latency_pair(
        self,
        fast_symbol: str,
        slow_symbol: str,
        **kwargs
    ) -> None:
        """Add a symbol pair for latency arbitrage monitoring."""
        self._latency_arb.add_pair(fast_symbol, slow_symbol, **kwargs)
    
    def process_quote(self, quote: Quote) -> CombinedSignal:
        """
        Process a quote update and generate combined signal.
        
        This is the main entry point for L1 data.
        
        Args:
            quote: New quote data
            
        Returns:
            Combined trading signal
        """
        symbol = quote.symbol
        
        # Calculate individual signals
        microprice_signal = self._microprice_calc.calculate_from_quote(quote)
        ofi_signal = self._ofi_calc.calculate_from_quote(quote)
        latency_signal = self._latency_arb.update_price(quote)
        
        # Combine signals
        combined = self._combine_signals(
            symbol=symbol,
            timestamp=quote.timestamp,
            microprice=microprice_signal,
            ofi=ofi_signal,
            latency_arb=latency_signal,
            spread_bps=quote.spread_bps,
        )
        
        # Store in history
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=1000)
        self._signal_history[symbol].append(combined)
        
        # Log significant signals
        if combined.should_trade(self._min_confidence):
            logger.log_signal(
                signal_name="combined",
                symbol=symbol,
                value=combined.signal_value,
                confidence=combined.confidence,
                microprice=microprice_signal.signal_value if microprice_signal else None,
                ofi=ofi_signal.signal_value if ofi_signal else None,
            )
        
        return combined
    
    def process_orderbook(self, book: OrderBook) -> CombinedSignal:
        """
        Process order book update and generate combined signal.
        
        Uses full order book for more accurate signals.
        
        Args:
            book: Updated order book
            
        Returns:
            Combined trading signal
        """
        symbol = book.symbol
        
        # Calculate signals from full book
        microprice_signal = self._microprice_calc.calculate_from_book(book)
        ofi_signal = self._ofi_calc.calculate(book)
        
        # Latency arb uses quotes, so create a quote from book
        quote = Quote(
            symbol=symbol,
            bid_price=book.best_bid_price,
            bid_size=book.best_bid.size if book.best_bid else 0,
            ask_price=book.best_ask_price,
            ask_size=book.best_ask.size if book.best_ask else 0,
            timestamp=book.timestamp,
        )
        latency_signal = self._latency_arb.update_price(quote)
        
        # Combine
        combined = self._combine_signals(
            symbol=symbol,
            timestamp=book.timestamp,
            microprice=microprice_signal,
            ofi=ofi_signal,
            latency_arb=latency_signal,
            spread_bps=book.spread_bps,
        )
        
        # Store
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=1000)
        self._signal_history[symbol].append(combined)
        
        return combined
    
    async def process_event(self, event: MarketDataEvent) -> Optional[CombinedSignal]:
        """
        Process a market data event.
        
        Async interface for event-driven architecture.
        """
        if event.event_type == MarketDataType.QUOTE:
            return self.process_quote(event.data)
        elif event.event_type == MarketDataType.ORDERBOOK:
            return self.process_orderbook(event.data)
        return None
    
    def _combine_signals(
        self,
        symbol: str,
        timestamp: datetime,
        microprice: Optional[MicropriceSignal],
        ofi: Optional[OFISignal],
        latency_arb: Optional[LatencyArbSignal],
        spread_bps: float = 0.0,
    ) -> CombinedSignal:
        """
        Combine individual signals into a final signal.
        
        Uses confidence-weighted combination:
        combined = Σ(weight_i * confidence_i * signal_i) / Σ(weight_i * confidence_i)
        """
        # Collect signals and weights
        signals = []
        weights = []
        confidences = []
        
        # Microprice
        if microprice:
            signals.append(microprice.signal_value)
            weights.append(self._weights.get("microprice", 0.4))
            confidences.append(microprice.confidence)
        
        # OFI
        if ofi:
            signals.append(ofi.signal_value)
            weights.append(self._weights.get("order_flow_imbalance", 0.35))
            confidences.append(ofi.confidence)
        
        # Latency arbitrage
        if latency_arb:
            signals.append(latency_arb.signal_value)
            weights.append(self._weights.get("latency_arbitrage", 0.25))
            confidences.append(latency_arb.confidence)
        
        # Calculate weighted combination
        if signals:
            signals_arr = np.array(signals)
            weights_arr = np.array(weights)
            conf_arr = np.array(confidences)
            
            # Confidence-weighted combination
            effective_weights = weights_arr * conf_arr
            weight_sum = np.sum(effective_weights)
            
            if weight_sum > 0:
                combined_signal = np.sum(effective_weights * signals_arr) / weight_sum
                combined_confidence = np.mean(conf_arr)  # Average confidence
            else:
                combined_signal = 0.0
                combined_confidence = 0.0
        else:
            combined_signal = 0.0
            combined_confidence = 0.0
        
        # Clip to valid range
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        combined_confidence = np.clip(combined_confidence, 0.0, 1.0)
        
        # Determine trading decision
        urgency, suggested_side, size_pct = self._determine_trading_decision(
            combined_signal, combined_confidence, spread_bps
        )
        
        # Get volatility estimate
        volatility = 0.0
        if ofi:
            volatility = ofi.volatility
        
        return CombinedSignal(
            symbol=symbol,
            timestamp=timestamp,
            signal_value=combined_signal,
            confidence=combined_confidence,
            microprice_signal=microprice.signal_value if microprice else None,
            ofi_signal=ofi.signal_value if ofi else None,
            latency_arb_signal=latency_arb.signal_value if latency_arb else None,
            microprice_confidence=microprice.confidence if microprice else 0.0,
            ofi_confidence=ofi.confidence if ofi else 0.0,
            latency_arb_confidence=latency_arb.confidence if latency_arb else 0.0,
            urgency=urgency,
            suggested_side=suggested_side,
            suggested_size_pct=size_pct,
            spread_bps=spread_bps,
            volatility=volatility,
        )
    
    def _determine_trading_decision(
        self,
        signal: float,
        confidence: float,
        spread_bps: float
    ) -> tuple:
        """
        Determine trading decision from combined signal.
        
        Returns:
            (urgency, side, size_pct)
        """
        # No trade if signal too weak or confidence too low
        if abs(signal) < 0.2 or confidence < self._min_confidence:
            return "LOW", "NONE", 0.0
        
        # Determine side
        if signal > 0:
            side = "BUY"
        else:
            side = "SELL"
        
        # Urgency based on signal strength and spread
        if abs(signal) > 0.7 and confidence > self._high_confidence:
            urgency = "HIGH"
        elif abs(signal) > 0.4:
            urgency = "NORMAL"
        else:
            urgency = "LOW"
        
        # Size based on confidence and signal strength
        # Higher confidence and stronger signal = larger position
        base_size = abs(signal) * confidence
        
        # Reduce size if spread is wide
        if spread_bps > 10:
            base_size *= 0.7
        elif spread_bps > 20:
            base_size *= 0.4
        
        # Cap at reasonable level
        size_pct = min(base_size, 0.5)
        
        return urgency, side, size_pct
    
    def get_signal_history(self, symbol: str) -> List[CombinedSignal]:
        """Get recent signal history for a symbol."""
        return list(self._signal_history.get(symbol, []))
    
    def get_signal_statistics(self, symbol: str) -> Dict[str, float]:
        """
        Get statistics on recent signals.
        
        Useful for monitoring signal quality and behavior.
        """
        history = self._signal_history.get(symbol, deque())
        if not history:
            return {}
        
        signals = [s.signal_value for s in history]
        confidences = [s.confidence for s in history]
        
        return {
            "count": len(signals),
            "mean_signal": np.mean(signals),
            "std_signal": np.std(signals),
            "mean_confidence": np.mean(confidences),
            "positive_ratio": sum(1 for s in signals if s > 0) / len(signals),
            "trade_ratio": sum(1 for s in history if s.should_trade()) / len(history),
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update signal weights dynamically.
        
        Useful for:
        - Regime adaptation
        - Learning from performance
        - A/B testing
        """
        self._weights.update(new_weights)
        self._normalize_weights()
        logger.info(
            f"Updated signal weights: {self._weights}",
            category=LogCategory.SIGNAL
        )
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset aggregator state."""
        if symbol:
            self._signal_history.pop(symbol, None)
            self._microprice_calc.reset(symbol)
            self._ofi_calc.reset(symbol)
        else:
            self._signal_history.clear()
            self._microprice_calc.reset()
            self._ofi_calc.reset()
            self._latency_arb.reset()
