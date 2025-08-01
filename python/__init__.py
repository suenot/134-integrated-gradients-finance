"""
Integrated Gradients for Finance - Python Implementation

This module provides tools for computing and analyzing integrated gradients
for financial trading models.
"""

from .integrated_gradients import IntegratedGradients
from .trading_model import TradingModel, TradingModelWithIG
from .data_loader import IGDataLoader
from .backtest import IGBacktester

__all__ = [
    "IntegratedGradients",
    "TradingModel",
    "TradingModelWithIG",
    "IGDataLoader",
    "IGBacktester",
]

__version__ = "0.1.0"
