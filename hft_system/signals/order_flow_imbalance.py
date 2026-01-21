"""
Order Flow Imbalance (OFI) Signal for HFT Research System
==========================================================

Order Flow Imbalance measures the net buying vs selling pressure
by analyzing changes in the order book over time.

THEORETICAL BACKGROUND:
=======================

OFI captures the CHANGE in order book state, not just the level.
This is important because:
1. New orders arriving = demand/supply
2. Cancellations = changing intentions
3. Executions = completed transactions

The OFI formula (from Cont, Kukanov, Stoikov 2014):

OFI_t = Σ (ΔQ^b - ΔQ^a)

where:
- ΔQ^b = change in bid quantity (positive = more buying interest)
- ΔQ^a = change in ask quantity (positive = more selling interest)

DECOMPOSITION:
- At best bid: if bid↑ then new buy orders; if bid↓ then cancels/fills
- At best ask: if ask↑ then new sell orders; if ask↓ then cancels/fills
- Price changes: bid moving up = aggressive buying lifted the offer

INTUITION:
- Positive OFI: More buying pressure than selling → Price likely to rise
- Negative OFI: More selling pressure than buying → Price likely to fall

KEY INSIGHT:
OFI is a FLOW measure (rate of change) not a STOCK measure (level).
This makes it more responsive to current market dynamics.

IMPLEMENTATION NOTES:
- Track order book state changes
- Handle price level changes (not just size)
- Normalize by recent volatility
- Consider multiple timeframes
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque, Tuple

from ..data.orderbook_simulator import OrderBook, PriceLevel, OrderSide
from ..data.market_data_feed import Quote
from ..infra.logging import get_logger, LogCategory
from ..infra.config import SignalConfig


logger = get_logger()


@dataclass
class OFISignal:
    """
    Output of Order Flow Imbalance calculation.
    
    Contains both raw OFI values and normalized signals
    suitable for trading decisions.
    """
    symbol: str
    timestamp: datetime
    
    # Core OFI values
    ofi_raw: float              # Raw OFI (sum of quantity changes)
    ofi_normalized: float       # Normalized by recent std
    signal_value: float         # -1 to +1 trading signal
    
    # Components
    bid_flow: float             # Net flow on bid side
    ask_flow: float             # Net flow on ask side
    
    # Confidence and quality
    confidence: float
    num_updates: int            # Updates in calculation window
    volatility: float           # Recent price volatility
    
    # Execution hint
    aggressor_bias: str         # "BUY", "SELL", or "NEUTRAL"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ofi_raw": self.ofi_raw,
            "ofi_normalized": self.ofi_normalized,
            "signal_value": self.signal_value,
            "confidence": self.confidence,
            "aggressor_bias": self.aggressor_bias,
        }


@dataclass
class BookState:
    """
    Snapshot of order book state for OFI calculation.
    """
    timestamp: datetime
    best_bid_price: float
    best_bid_size: int
    best_ask_price: float
    best_ask_size: int
    mid_price: float


class OrderFlowImbalanceCalculator:
    """
    Calculates Order Flow Imbalance signals.
    
    OFI is computed by tracking changes in the order book:
    - Increases in bid size = buying pressure
    - Increases in ask size = selling pressure
    - Price level changes indicate aggressive orders
    """
    
    def __init__(self, config: SignalConfig):
        """
        Initialize OFI calculator.
        
        Args:
            config: Signal configuration
        """
        self._config = config
        
        # State tracking per symbol
        self._previous_states: Dict[str, BookState] = {}
        
        # OFI history for normalization
        self._ofi_history: Dict[str, Deque[float]] = {}
        self._price_history: Dict[str, Deque[float]] = {}
        
        # Rolling statistics
        self._ofi_means: Dict[str, float] = {}
        self._ofi_stds: Dict[str, float] = {}
        self._volatilities: Dict[str, float] = {}
        
        # Parameters
        self._window_size = config.ofi_window_ticks
        self._norm_window = config.ofi_normalization_window
    
    def calculate(self, book: OrderBook) -> Optional[OFISignal]:
        """
        Calculate OFI signal from order book update.
        
        The core OFI calculation follows Cont et al. (2014):
        
        OFI = Σ (e_n^B - e_n^A)
        
        where:
        e_n^B = bid contribution (positive = buying)
        e_n^A = ask contribution (positive = selling)
        
        Returns None if this is the first update (need previous state).
        """
        symbol = book.symbol
        
        # Initialize if needed
        if symbol not in self._ofi_history:
            self._ofi_history[symbol] = deque(maxlen=self._norm_window)
            self._price_history[symbol] = deque(maxlen=self._norm_window)
            self._ofi_means[symbol] = 0.0
            self._ofi_stds[symbol] = 1.0
            self._volatilities[symbol] = 0.0001
        
        # Create current state
        current_state = BookState(
            timestamp=book.timestamp,
            best_bid_price=book.best_bid_price,
            best_bid_size=book.best_bid.size if book.best_bid else 0,
            best_ask_price=book.best_ask_price,
            best_ask_size=book.best_ask.size if book.best_ask else 0,
            mid_price=book.mid_price,
        )
        
        # Check if we have previous state
        prev_state = self._previous_states.get(symbol)
        
        if prev_state is None:
            # First update - store state and return None
            self._previous_states[symbol] = current_state
            self._price_history[symbol].append(current_state.mid_price)
            return None
        
        # Calculate OFI components
        bid_flow, ask_flow = self._calculate_flow_components(
            prev_state, current_state
        )
        
        ofi_raw = bid_flow - ask_flow
        
        # Update history
        self._ofi_history[symbol].append(ofi_raw)
        self._price_history[symbol].append(current_state.mid_price)
        
        # Update volatility
        self._update_volatility(symbol)
        
        # Normalize OFI by recent volatility
        ofi_normalized = self._normalize_ofi(symbol, ofi_raw)
        
        # Update rolling statistics
        self._update_statistics(symbol)
        
        # Convert to trading signal
        signal_value = np.tanh(ofi_normalized)
        
        # Determine aggressor bias
        aggressor_bias = self._determine_aggressor(bid_flow, ask_flow)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            symbol, ofi_normalized, bid_flow, ask_flow, book
        )
        
        # Store current state for next calculation
        self._previous_states[symbol] = current_state
        
        return OFISignal(
            symbol=symbol,
            timestamp=book.timestamp,
            ofi_raw=ofi_raw,
            ofi_normalized=ofi_normalized,
            signal_value=signal_value,
            bid_flow=bid_flow,
            ask_flow=ask_flow,
            confidence=confidence,
            num_updates=len(self._ofi_history[symbol]),
            volatility=self._volatilities[symbol],
            aggressor_bias=aggressor_bias,
        )
    
    def calculate_from_quote(self, quote: Quote) -> Optional[OFISignal]:
        """
        Calculate OFI from L1 quote (simplified).
        
        Less accurate than full book but works with free data.
        """
        symbol = quote.symbol
        
        # Initialize if needed
        if symbol not in self._ofi_history:
            self._ofi_history[symbol] = deque(maxlen=self._norm_window)
            self._price_history[symbol] = deque(maxlen=self._norm_window)
            self._ofi_means[symbol] = 0.0
            self._ofi_stds[symbol] = 1.0
            self._volatilities[symbol] = 0.0001
        
        # Create current state
        current_state = BookState(
            timestamp=quote.timestamp,
            best_bid_price=quote.bid_price,
            best_bid_size=quote.bid_size,
            best_ask_price=quote.ask_price,
            best_ask_size=quote.ask_size,
            mid_price=quote.mid_price,
        )
        
        prev_state = self._previous_states.get(symbol)
        
        if prev_state is None:
            self._previous_states[symbol] = current_state
            self._price_history[symbol].append(current_state.mid_price)
            return None
        
        # Calculate flow
        bid_flow, ask_flow = self._calculate_flow_components(
            prev_state, current_state
        )
        
        ofi_raw = bid_flow - ask_flow
        
        # Update history
        self._ofi_history[symbol].append(ofi_raw)
        self._price_history[symbol].append(current_state.mid_price)
        
        # Update volatility
        self._update_volatility(symbol)
        
        # Normalize and convert
        ofi_normalized = self._normalize_ofi(symbol, ofi_raw)
        self._update_statistics(symbol)
        signal_value = np.tanh(ofi_normalized)
        aggressor_bias = self._determine_aggressor(bid_flow, ask_flow)
        
        # Lower confidence for L1 data
        confidence = self._calculate_confidence_l1(
            symbol, ofi_normalized, bid_flow, ask_flow, quote
        )
        
        self._previous_states[symbol] = current_state
        
        return OFISignal(
            symbol=symbol,
            timestamp=quote.timestamp,
            ofi_raw=ofi_raw,
            ofi_normalized=ofi_normalized,
            signal_value=signal_value,
            bid_flow=bid_flow,
            ask_flow=ask_flow,
            confidence=confidence,
            num_updates=len(self._ofi_history[symbol]),
            volatility=self._volatilities[symbol],
            aggressor_bias=aggressor_bias,
        )
    
    def _calculate_flow_components(
        self,
        prev: BookState,
        curr: BookState
    ) -> Tuple[float, float]:
        """
        Calculate bid and ask flow components.
        
        This implements the OFI decomposition from Cont et al.:
        
        Bid contribution (e^B):
        - If bid price goes UP: all previous bid size + new size (aggressive buy)
        - If bid price unchanged: change in size
        - If bid price goes DOWN: -all current bid size (sell pressure)
        
        Ask contribution (e^A):
        - If ask price goes DOWN: all previous ask size + new size (aggressive sell)
        - If ask price unchanged: change in size  
        - If ask price goes UP: -all current ask size (buy pressure lifted offers)
        """
        bid_flow = 0.0
        ask_flow = 0.0
        
        # Bid side analysis
        if curr.best_bid_price > prev.best_bid_price:
            # Bid improved (aggressive buying lifted the offer?)
            # This is bullish - treat as positive bid flow
            bid_flow = prev.best_bid_size + curr.best_bid_size
            
        elif curr.best_bid_price < prev.best_bid_price:
            # Bid dropped (bids got hit by sellers)
            # This is bearish - negative bid flow
            bid_flow = -curr.best_bid_size
            
        else:
            # Price unchanged - just size change
            bid_flow = curr.best_bid_size - prev.best_bid_size
        
        # Ask side analysis
        if curr.best_ask_price < prev.best_ask_price:
            # Ask improved (aggressive selling?)
            # This is bearish - positive ask flow (selling pressure)
            ask_flow = prev.best_ask_size + curr.best_ask_size
            
        elif curr.best_ask_price > prev.best_ask_price:
            # Ask moved up (offers got lifted by buyers)
            # This is bullish - negative ask flow
            ask_flow = -curr.best_ask_size
            
        else:
            # Price unchanged - just size change
            ask_flow = curr.best_ask_size - prev.best_ask_size
        
        return bid_flow, ask_flow
    
    def _normalize_ofi(self, symbol: str, ofi_raw: float) -> float:
        """
        Normalize OFI by recent volatility and typical OFI magnitude.
        
        This makes the signal comparable across:
        - Different symbols (with different typical volumes)
        - Different market regimes (high/low vol)
        """
        volatility = self._volatilities.get(symbol, 0.0001)
        std = self._ofi_stds.get(symbol, 1.0)
        
        # Normalize by std of recent OFI values
        if std > 0.01:
            normalized = ofi_raw / std
        else:
            normalized = ofi_raw / 100  # Fallback scaling
        
        # Scale by inverse volatility (OFI more meaningful in low vol)
        # But cap to avoid extreme values
        vol_scale = min(0.01 / max(volatility, 0.0001), 5.0)
        
        return normalized * vol_scale
    
    def _update_statistics(self, symbol: str) -> None:
        """Update rolling OFI statistics."""
        history = self._ofi_history[symbol]
        if len(history) < 5:
            return
        
        # Simple rolling std
        recent = list(history)[-self._window_size:]
        self._ofi_means[symbol] = np.mean(recent)
        self._ofi_stds[symbol] = max(np.std(recent), 0.01)
    
    def _update_volatility(self, symbol: str) -> None:
        """Update rolling price volatility."""
        prices = self._price_history[symbol]
        if len(prices) < 10:
            return
        
        # Calculate returns
        recent_prices = list(prices)[-50:]
        returns = np.diff(np.log(np.array(recent_prices) + 0.0001))
        
        self._volatilities[symbol] = max(np.std(returns), 0.0001)
    
    def _determine_aggressor(self, bid_flow: float, ask_flow: float) -> str:
        """
        Determine likely aggressor side.
        
        In market microstructure:
        - Aggressive buyers cross the spread to hit asks
        - Aggressive sellers cross to hit bids
        
        The net flow tells us which side is more aggressive.
        """
        net_flow = bid_flow - ask_flow
        
        # Threshold for significance
        threshold = 50  # Shares
        
        if net_flow > threshold:
            return "BUY"
        elif net_flow < -threshold:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def _calculate_confidence(
        self,
        symbol: str,
        ofi_normalized: float,
        bid_flow: float,
        ask_flow: float,
        book: OrderBook
    ) -> float:
        """
        Calculate confidence in the OFI signal.
        
        Higher confidence when:
        1. Signal is strong (large normalized value)
        2. Bid and ask flow agree (both point same direction)
        3. Spread is tight
        4. Signal is consistent with recent history
        """
        confidence = 0.5
        
        # Factor 1: Signal strength
        strength = min(abs(ofi_normalized), 2.0) / 2.0
        confidence += strength * 0.2
        
        # Factor 2: Flow agreement
        # If bid_flow > 0 and ask_flow < 0, both indicate buying
        if (bid_flow > 0 and ask_flow <= 0) or (bid_flow <= 0 and ask_flow > 0):
            confidence += 0.15
        elif (bid_flow > 0 and ask_flow > 0) or (bid_flow < 0 and ask_flow < 0):
            # Conflicting signals
            confidence -= 0.1
        
        # Factor 3: Spread
        if book.spread_bps < 5:
            confidence += 0.1
        elif book.spread_bps > 15:
            confidence -= 0.1
        
        # Factor 4: Consistency with history
        history = self._ofi_history[symbol]
        if len(history) >= 3:
            recent = list(history)[-3:]
            if all(r > 0 for r in recent) or all(r < 0 for r in recent):
                confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_confidence_l1(
        self,
        symbol: str,
        ofi_normalized: float,
        bid_flow: float,
        ask_flow: float,
        quote: Quote
    ) -> float:
        """Lower confidence calculation for L1 data."""
        confidence = 0.4  # Lower base
        
        # Signal strength
        strength = min(abs(ofi_normalized), 2.0) / 2.0
        confidence += strength * 0.15
        
        # Flow agreement
        if (bid_flow > 0 and ask_flow <= 0) or (bid_flow <= 0 and ask_flow > 0):
            confidence += 0.1
        
        # Spread
        if quote.spread_bps < 10:
            confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_cumulative_ofi(
        self,
        symbol: str,
        window: int = 10
    ) -> float:
        """
        Get cumulative OFI over a window.
        
        Useful for detecting sustained pressure.
        """
        history = self._ofi_history.get(symbol, deque())
        if len(history) < window:
            return sum(history)
        return sum(list(history)[-window:])
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset calculator state."""
        if symbol:
            self._previous_states.pop(symbol, None)
            self._ofi_history.pop(symbol, None)
            self._price_history.pop(symbol, None)
            self._ofi_means.pop(symbol, None)
            self._ofi_stds.pop(symbol, None)
            self._volatilities.pop(symbol, None)
        else:
            self._previous_states.clear()
            self._ofi_history.clear()
            self._price_history.clear()
            self._ofi_means.clear()
            self._ofi_stds.clear()
            self._volatilities.clear()
