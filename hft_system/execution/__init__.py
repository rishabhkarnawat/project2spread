"""
Execution module for HFT Research System.

Provides order routing and execution simulation:
- Smart order routing
- Execution simulation with realistic effects
"""

from .smart_order_router import (
    SmartOrderRouter,
    Order,
    OrderType,
    OrderStatus,
    OrderSideEnum,
)

from .execution_simulator import (
    ExecutionSimulator,
    ExecutionResult,
)

__all__ = [
    # Router
    "SmartOrderRouter",
    "Order",
    "OrderType",
    "OrderStatus",
    "OrderSideEnum",
    # Simulator
    "ExecutionSimulator",
    "ExecutionResult",
]
