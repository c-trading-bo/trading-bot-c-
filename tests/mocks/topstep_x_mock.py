#!/usr/bin/env python3
"""
Mock TopstepX SDK for Testing

This module provides mock implementations for the TopstepX SDK components
for use in testing and development environments only.
"""

import asyncio
import uuid
from typing import Dict, Any


class MockTradingSuite:
    def __init__(self, instruments, **kwargs):
        self.instruments = instruments
        self._connected = True
        
    @classmethod
    async def create(cls, instruments, **kwargs):
        # Simulate SDK initialization behavior
        await asyncio.sleep(0.1)  # Simulate connection delay
        return cls(instruments, **kwargs)
        
    def __getitem__(self, instrument):
        return MockInstrument(instrument)
        
    async def get_stats(self):
        return {
            "total_trades": 42,
            "win_rate": 65.5,
            "total_pnl": 1250.75,
            "max_drawdown": -150.25
        }
        
    async def get_risk_metrics(self):
        return {
            "max_risk_percent": 1.0,
            "current_risk": 0.15,
            "available_buying_power": 50000.0
        }
        
    async def get_portfolio_status(self):
        return {
            "account_value": 50000.0,
            "buying_power": 45000.0,
            "day_pnl": 125.50
        }
        
    async def disconnect(self):
        self._connected = False
        
    def managed_trade(self, max_risk_percent=0.01):
        return MockManagedTradeContext(max_risk_percent)


class MockInstrument:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = MockData(symbol)
        self.orders = MockOrders(symbol)
        self.positions = MockPositions()
        
    async def get_position(self):
        return {
            "size": 0,
            "average_price": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0
        }


class MockData:
    def __init__(self, symbol):
        self.symbol = symbol
        
    async def get_current_price(self):
        # Realistic prices for MNQ and ES
        prices = {
            'MNQ': 18500.00,
            'ES': 4500.00,
            'NQ': 18500.00,
            'RTY': 2100.00,
            'YM': 35000.00
        }
        return prices.get(self.symbol, 100.00)


class MockOrders:
    def __init__(self, symbol):
        self.symbol = symbol
        
    async def place_bracket_order(self, side, quantity, stop_loss, take_profit):
        return {
            "id": str(uuid.uuid4()),
            "entry_order_id": str(uuid.uuid4()),
            "stop_order_id": str(uuid.uuid4()),
            "target_order_id": str(uuid.uuid4()),
            "status": "accepted"
        }


class MockPositions:
    def __init__(self):
        self.quantity = 0
        self.avg_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0


class MockManagedTradeContext:
    def __init__(self, max_risk_percent):
        self.max_risk_percent = max_risk_percent
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass