"""
High-Performance Logging for HFT Research System
=================================================

This module implements logging optimized for trading systems:
- Async logging to prevent blocking the hot path
- Structured logging for machine parsing
- Latency measurement integration
- Separate streams for different log types

PRODUCTION HFT LOGGING CONSIDERATIONS:
======================================

1. LATENCY IMPACT:
   - Standard Python logging can add 1-10 microseconds per call
   - In production HFT, logging is often:
     * Disabled on hot path entirely
     * Written to shared memory, processed by separate core
     * Binary format (not human-readable) for speed
     * Post-processed for analysis

2. LOG TYPES IN REAL SYSTEMS:
   - Audit logs: Regulatory requirement, every order/fill
   - Performance logs: Latency measurements, throughput
   - Risk logs: Position changes, limit breaches
   - Debug logs: Development only, never in production hot path

3. STORAGE:
   - Time-series databases (InfluxDB, TimescaleDB)
   - Columnar formats for analysis (Parquet)
   - Real-time streaming (Kafka) for monitoring

This implementation uses Python's asyncio-compatible logging
with structured format for later analysis.
"""

import asyncio
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Deque
from contextlib import contextmanager
import json
import threading
from queue import Queue


class LogCategory(Enum):
    """
    Log categories for filtering and routing.
    
    In production, these would route to different:
    - Storage backends
    - Retention policies
    - Alert systems
    """
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    EXECUTION = "execution"
    RISK = "risk"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    AUDIT = "audit"


@dataclass
class LatencyMeasurement:
    """
    Captures latency for a specific operation.
    
    In real HFT, latency is measured with:
    - Hardware timestamps (NIC, FPGA)
    - Kernel bypass (DPDK, Solarflare OpenOnload)
    - Sub-microsecond precision
    
    We use Python's time.perf_counter_ns() which provides
    nanosecond resolution but ~100ns accuracy on most systems.
    """
    operation: str
    start_ns: int
    end_ns: int = 0
    category: LogCategory = LogCategory.PERFORMANCE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns
    
    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0
    
    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ns": self.duration_ns,
            "duration_us": self.duration_us,
            "duration_ms": self.duration_ms,
            "category": self.category.value,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }


class LatencyTracker:
    """
    Tracks latency statistics for different operations.
    
    Maintains rolling statistics without storing all measurements
    to minimize memory impact.
    """
    
    def __init__(self, window_size: int = 10000):
        self._window_size = window_size
        self._measurements: Dict[str, Deque[int]] = {}
        self._lock = threading.Lock()
    
    def record(self, measurement: LatencyMeasurement) -> None:
        """Record a latency measurement."""
        with self._lock:
            if measurement.operation not in self._measurements:
                self._measurements[measurement.operation] = deque(maxlen=self._window_size)
            self._measurements[measurement.operation].append(measurement.duration_ns)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get latency statistics for an operation.
        
        Returns percentiles which are critical for HFT:
        - p50: Typical latency
        - p99: Tail latency (what matters for consistency)
        - p99.9: Extreme tail (regulatory/risk concern)
        """
        with self._lock:
            if operation not in self._measurements or not self._measurements[operation]:
                return {}
            
            measurements = sorted(self._measurements[operation])
            n = len(measurements)
            
            return {
                "count": n,
                "min_ns": measurements[0],
                "max_ns": measurements[-1],
                "mean_ns": sum(measurements) / n,
                "p50_ns": measurements[n // 2],
                "p99_ns": measurements[int(n * 0.99)] if n >= 100 else measurements[-1],
                "p999_ns": measurements[int(n * 0.999)] if n >= 1000 else measurements[-1],
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations."""
        with self._lock:
            return {op: self.get_stats(op) for op in self._measurements.keys()}


class AsyncLogHandler(logging.Handler):
    """
    Async-compatible log handler that doesn't block the event loop.
    
    Uses a background thread to write logs, similar to production
    systems that offload logging to prevent hot-path impact.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue: Queue = Queue(maxsize=100000)
        self._shutdown = False
        self._worker = threading.Thread(target=self._process_logs, daemon=True)
        self._worker.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue log record for async processing."""
        try:
            # Non-blocking put, drop if queue full (production behavior)
            self._queue.put_nowait(record)
        except Exception:
            # In production, would increment a "dropped logs" counter
            pass
    
    def _process_logs(self) -> None:
        """Background worker to process queued logs."""
        while not self._shutdown:
            try:
                record = self._queue.get(timeout=0.1)
                # Format and write - in production, would write to file/network
                msg = self.format(record)
                print(msg, file=sys.stderr, flush=True)
            except Exception:
                continue
    
    def close(self) -> None:
        """Shutdown the handler."""
        self._shutdown = True
        self._worker.join(timeout=1.0)
        super().close()


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter for machine parsing.
    
    Structured logs enable:
    - Easy querying in log aggregation systems (ELK, Splunk)
    - Automated alerting based on field values
    - Performance analysis across time ranges
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add any extra fields attached to the record
        if hasattr(record, "category"):
            log_data["category"] = record.category
        if hasattr(record, "symbol"):
            log_data["symbol"] = record.symbol
        if hasattr(record, "latency_ns"):
            log_data["latency_ns"] = record.latency_ns
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class HFTLogger:
    """
    High-performance logger for HFT system components.
    
    Provides:
    - Categorized logging
    - Latency tracking integration
    - Context managers for timing
    - Structured output
    """
    
    _instance: Optional['HFTLogger'] = None
    _latency_tracker: LatencyTracker = LatencyTracker()
    
    def __init__(self, name: str = "hft_system", level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # Remove existing handlers
        self._logger.handlers = []
        
        # Add async handler with structured formatter
        handler = AsyncLogHandler()
        handler.setFormatter(StructuredFormatter())
        self._logger.addHandler(handler)
        
        # Also add a simple console handler for debugging
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        console.setLevel(logging.DEBUG)
        self._logger.addHandler(console)
    
    @classmethod
    def get_instance(cls) -> 'HFTLogger':
        """Singleton pattern for global logger access."""
        if cls._instance is None:
            cls._instance = HFTLogger()
        return cls._instance
    
    @classmethod
    def get_latency_tracker(cls) -> LatencyTracker:
        """Access the shared latency tracker."""
        return cls._latency_tracker
    
    def log(
        self,
        level: int,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        symbol: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a message with HFT-specific context.
        """
        extra = {
            "category": category.value,
            "extra_data": kwargs
        }
        if symbol:
            extra["symbol"] = symbol
        
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        self.log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self.log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        self.log(logging.CRITICAL, message, **kwargs)
    
    @contextmanager
    def measure_latency(
        self,
        operation: str,
        category: LogCategory = LogCategory.PERFORMANCE,
        log_level: int = logging.DEBUG,
        **metadata
    ):
        """
        Context manager to measure and log operation latency.
        
        Usage:
            with logger.measure_latency("signal_computation", symbol="AAPL"):
                # ... compute signal ...
        
        This is critical for:
        - Identifying performance bottlenecks
        - Ensuring latency budgets are met
        - Detecting degradation over time
        """
        measurement = LatencyMeasurement(
            operation=operation,
            start_ns=time.perf_counter_ns(),
            category=category,
            metadata=metadata
        )
        
        try:
            yield measurement
        finally:
            measurement.end_ns = time.perf_counter_ns()
            self._latency_tracker.record(measurement)
            
            self.log(
                log_level,
                f"{operation} completed in {measurement.duration_us:.2f}μs",
                category=category,
                latency_ns=measurement.duration_ns,
                **metadata
            )
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: str,
        **kwargs
    ) -> None:
        """
        Log trade execution - audit trail requirement.
        
        In production, this would:
        - Write to immutable audit log
        - Include microsecond timestamps
        - Be cryptographically signed
        - Replicate to compliance systems
        """
        self.log(
            logging.INFO,
            f"TRADE: {side} {quantity} {symbol} @ {price:.4f}",
            category=LogCategory.AUDIT,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            **kwargs
        )
    
    def log_signal(
        self,
        signal_name: str,
        symbol: str,
        value: float,
        confidence: float,
        **kwargs
    ) -> None:
        """Log signal generation for analysis."""
        self.log(
            logging.DEBUG,
            f"SIGNAL: {signal_name} {symbol} = {value:.6f} (conf: {confidence:.2f})",
            category=LogCategory.SIGNAL,
            symbol=symbol,
            signal_name=signal_name,
            signal_value=value,
            confidence=confidence,
            **kwargs
        )
    
    def log_risk_event(
        self,
        event_type: str,
        message: str,
        severity: str = "WARNING",
        **kwargs
    ) -> None:
        """
        Log risk management events.
        
        Risk events are high priority and may trigger:
        - Alerts to risk managers
        - Automatic position reduction
        - System shutdown in extreme cases
        """
        level = logging.WARNING if severity == "WARNING" else logging.CRITICAL
        self.log(
            level,
            f"RISK [{event_type}]: {message}",
            category=LogCategory.RISK,
            event_type=event_type,
            severity=severity,
            **kwargs
        )


# Global logger instance
logger = HFTLogger.get_instance()


def get_logger() -> HFTLogger:
    """Get the global HFT logger instance."""
    return logger


def get_latency_stats() -> Dict[str, Dict[str, float]]:
    """Get latency statistics for all tracked operations."""
    return HFTLogger.get_latency_tracker().get_all_stats()
