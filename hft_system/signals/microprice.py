"""
Microprice Signal for HFT Research System
==========================================

The microprice is one of the most important signals in market microstructure.
It uses the order book imbalance to predict short-term price direction.

THEORETICAL BACKGROUND:
=======================

The traditional mid-price treats bid and ask equally:
    mid = (bid + ask) / 2

But this ignores valuable information: the SIZE at each level.

The microprice incorporates size imbalance:
    microprice = (bid_size * ask + ask_size * bid) / (bid_size + ask_size)

INTUITION:
- If bid_size >> ask_size: More buyers than sellers
- Price likely to move UP toward the ask
- Microprice > mid (closer to ask)

- If ask_size >> bid_size: More sellers than buyers  
- Price likely to move DOWN toward the bid
- Microprice < mid (closer to bid)

EMPIRICAL EVIDENCE:
- Microprice is the best linear predictor of next trade price
- Works on millisecond to second horizons
- Alpha decays quickly (competition)
- More effective in liquid markets

VARIATIONS IMPLEMENTED:
1. Basic microprice (Level 1 only)
2. Weighted microprice (multiple levels)
3. Exponential decay microprice (time-weighted)
4. Normalized microprice signal (z-score)

PRODUCTION CONSIDERATIONS:
- Computation must be < 1 microsecond
- Use SIMD instructions for vectorization
- Pre-compute normalization factors
- Cache results at each price update
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque, Tuple

from ..data.orderbook_simulator import OrderBook, PriceLevel
from ..data.market_data_feed import Quote
from ..infra.logging import get_logger, LogCategory
from ..infra.config import SignalConfig


logger = get_logger()


@dataclass
class MicropriceSignal:
    """
    Output of microprice signal calculation.
    
    Contains the raw signal value plus metadata for
    signal quality assessment and position sizing.
    """
    symbol: str
    timestamp: datetime
    
    # Core values
    microprice: float      # The microprice itself
    mid_price: float       # Standard mid for comparison
    signal_value: float    # Normalized signal (-1 to +1)
    
    # Confidence and quality
    confidence: float      # 0 to 1, based on signal strength and stability
    spread_bps: float      # Current spread (affects execution)
    imbalance: float       # Raw imbalance ratio
    
    # Metadata
    levels_used: int       # How many book levels contributed
    lookback_ticks: int    # Ticks in normalization window
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "microprice": self.microprice,
            "mid_price": self.mid_price,
            "signal_value": self.signal_value,
            "confidence": self.confidence,
            "spread_bps": self.spread_bps,
            "imbalance": self.imbalance,
        }


class MicropriceCalculator:
    """
    Calculates microprice signals from order book data.
    
    This is a core alpha signal in HFT, predicting short-term
    price movements based on order book imbalance.
    """
    
    def __init__(self, config: SignalConfig):
        """
        Initialize microprice calculator.
        
        Args:
            config: Signal configuration parameters
        """
        self._config = config
        
        # History for normalization
        self._signal_history: Dict[str, Deque[float]] = {}
        self._microprice_history: Dict[str, Deque[float]] = {}
        
        # Rolling statistics
        self._signal_means: Dict[str, float] = {}
        self._signal_stds: Dict[str, float] = {}
        
        # Exponential decay factors
        self._decay_factor = config.microprice_decay_factor
        self._lookback = config.microprice_lookback_ticks
    
    def calculate_from_book(self, book: OrderBook) -> MicropriceSignal:
        """
        Calculate microprice signal from full order book.
        
        Uses multiple levels for more robust estimation.
        """
        symbol = book.symbol
        
        # Initialize history if needed
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=self._lookback)
            self._microprice_history[symbol] = deque(maxlen=self._lookback)
            self._signal_means[symbol] = 0.0
            self._signal_stds[symbol] = 1.0
        
        # Calculate weighted microprice across multiple levels
        microprice = self._calculate_weighted_microprice(book)
        mid_price = book.mid_price
        
        # Raw imbalance
        imbalance = book.top_of_book_imbalance
        
        # Raw signal: deviation from mid
        if mid_price > 0:
            raw_signal = (microprice - mid_price) / mid_price * 10000  # In bps
        else:
            raw_signal = 0.0
        
        # Update history
        self._signal_history[symbol].append(raw_signal)
        self._microprice_history[symbol].append(microprice)
        
        # Update rolling statistics
        self._update_statistics(symbol)
        
        # Normalize signal (z-score)
        if self._signal_stds[symbol] > 0.01:
            normalized_signal = (raw_signal - self._signal_means[symbol]) / self._signal_stds[symbol]
        else:
            normalized_signal = 0.0
        
        # Clip to reasonable range
        normalized_signal = np.clip(normalized_signal, -3.0, 3.0)
        
        # Convert to -1 to +1 range using tanh
        signal_value = np.tanh(normalized_signal)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            symbol, book, signal_value, imbalance
        )
        
        return MicropriceSignal(
            symbol=symbol,
            timestamp=book.timestamp,
            microprice=microprice,
            mid_price=mid_price,
            signal_value=signal_value,
            confidence=confidence,
            spread_bps=book.spread_bps,
            imbalance=imbalance,
            levels_used=min(len(book.bids), len(book.asks)),
            lookback_ticks=len(self._signal_history[symbol]),
        )
    
    def calculate_from_quote(self, quote: Quote) -> MicropriceSignal:
        """
        Calculate microprice signal from L1 quote only.
        
        Simpler calculation when full book not available.
        """
        symbol = quote.symbol
        
        # Initialize history if needed
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=self._lookback)
            self._microprice_history[symbol] = deque(maxlen=self._lookback)
            self._signal_means[symbol] = 0.0
            self._signal_stds[symbol] = 1.0
        
        # L1 microprice
        total_size = quote.bid_size + quote.ask_size
        if total_size > 0:
            microprice = (
                quote.bid_size * quote.ask_price +
                quote.ask_size * quote.bid_price
            ) / total_size
        else:
            microprice = quote.mid_price
        
        mid_price = quote.mid_price
        
        # Raw signal
        if mid_price > 0:
            raw_signal = (microprice - mid_price) / mid_price * 10000
        else:
            raw_signal = 0.0
        
        # Update history
        self._signal_history[symbol].append(raw_signal)
        self._microprice_history[symbol].append(microprice)
        
        # Update statistics
        self._update_statistics(symbol)
        
        # Normalize
        if self._signal_stds[symbol] > 0.01:
            normalized_signal = (raw_signal - self._signal_means[symbol]) / self._signal_stds[symbol]
        else:
            normalized_signal = 0.0
        
        normalized_signal = np.clip(normalized_signal, -3.0, 3.0)
        signal_value = np.tanh(normalized_signal)
        
        # Confidence (lower for L1 only)
        confidence = self._calculate_confidence_from_quote(
            symbol, quote, signal_value
        )
        
        return MicropriceSignal(
            symbol=symbol,
            timestamp=quote.timestamp,
            microprice=microprice,
            mid_price=mid_price,
            signal_value=signal_value,
            confidence=confidence,
            spread_bps=quote.spread_bps,
            imbalance=quote.imbalance,
            levels_used=1,
            lookback_ticks=len(self._signal_history[symbol]),
        )
    
    def _calculate_weighted_microprice(self, book: OrderBook) -> float:
        """
        Calculate microprice using multiple book levels with decay.
        
        Deeper levels get less weight because:
        1. They're further from where trades happen
        2. They're more likely to be canceled
        3. They may be iceberg/hidden orders
        
        Formula:
        microprice = Σ(w_i * (bid_size_i * ask_i + ask_size_i * bid_i)) / Σ(w_i * (bid_size_i + ask_size_i))
        
        where w_i = decay_factor ^ i (exponential decay)
        """
        if not book.bids or not book.asks:
            return book.mid_price
        
        numerator = 0.0
        denominator = 0.0
        
        n_levels = min(len(book.bids), len(book.asks), 5)  # Use top 5 levels
        
        for i in range(n_levels):
            weight = self._decay_factor ** i
            
            bid_level = book.bids[i]
            ask_level = book.asks[i]
            
            # Weighted contribution from this level
            level_num = weight * (
                bid_level.size * ask_level.price +
                ask_level.size * bid_level.price
            )
            level_denom = weight * (bid_level.size + ask_level.size)
            
            numerator += level_num
            denominator += level_denom
        
        if denominator > 0:
            return numerator / denominator
        return book.mid_price
    
    def _update_statistics(self, symbol: str) -> None:
        """
        Update rolling mean and standard deviation.
        
        Uses exponential moving average for responsiveness.
        """
        history = self._signal_history[symbol]
        if len(history) < 10:
            return
        
        # EMA parameters
        span = min(len(history), self._lookback)
        alpha = 2.0 / (span + 1)
        
        # Update mean with EMA
        new_value = history[-1]
        old_mean = self._signal_means[symbol]
        self._signal_means[symbol] = alpha * new_value + (1 - alpha) * old_mean
        
        # Update variance with EMA
        # Using Welford's method adapted for EMA
        deviation = new_value - self._signal_means[symbol]
        old_var = self._signal_stds[symbol] ** 2
        new_var = (1 - alpha) * (old_var + alpha * deviation ** 2)
        self._signal_stds[symbol] = np.sqrt(max(new_var, 0.01))
    
    def _calculate_confidence(
        self,
        symbol: str,
        book: OrderBook,
        signal_value: float,
        imbalance: float
    ) -> float:
        """
        Calculate confidence score for the signal.
        
        Confidence is higher when:
        1. Signal is consistent over time (not noisy)
        2. Spread is tight (execution is cheap)
        3. Book depth is adequate (liquidity available)
        4. Signal magnitude is significant
        
        Confidence is lower when:
        1. Signal is oscillating
        2. Spread is wide
        3. Book is thin
        4. Signal is weak
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Signal magnitude (stronger = more confident)
        magnitude_factor = min(abs(signal_value), 1.0) * 0.2
        confidence += magnitude_factor
        
        # Factor 2: Spread (tighter = more confident)
        if book.spread_bps < 5:
            spread_factor = 0.2
        elif book.spread_bps < 10:
            spread_factor = 0.1
        else:
            spread_factor = -0.1
        confidence += spread_factor
        
        # Factor 3: Imbalance consistency with signal
        # If imbalance and signal agree, more confident
        if imbalance * signal_value > 0:  # Same sign
            agreement_factor = min(abs(imbalance), 0.5) * 0.2
            confidence += agreement_factor
        else:
            confidence -= 0.1
        
        # Factor 4: Signal stability (not oscillating)
        if len(self._signal_history[symbol]) >= 5:
            recent = list(self._signal_history[symbol])[-5:]
            sign_changes = sum(1 for i in range(1, len(recent)) if recent[i] * recent[i-1] < 0)
            if sign_changes <= 1:
                confidence += 0.1
            elif sign_changes >= 3:
                confidence -= 0.2
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_confidence_from_quote(
        self,
        symbol: str,
        quote: Quote,
        signal_value: float
    ) -> float:
        """
        Calculate confidence from L1 quote (simpler version).
        """
        confidence = 0.4  # Lower base for L1 only
        
        # Magnitude
        confidence += min(abs(signal_value), 1.0) * 0.15
        
        # Spread
        if quote.spread_bps < 10:
            confidence += 0.15
        
        # Imbalance
        if quote.imbalance * signal_value > 0:
            confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_signal_history(self, symbol: str) -> List[float]:
        """Get recent signal history for analysis."""
        return list(self._signal_history.get(symbol, []))
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset signal state."""
        if symbol:
            self._signal_history.pop(symbol, None)
            self._microprice_history.pop(symbol, None)
            self._signal_means.pop(symbol, None)
            self._signal_stds.pop(symbol, None)
        else:
            self._signal_history.clear()
            self._microprice_history.clear()
            self._signal_means.clear()
            self._signal_stds.clear()
