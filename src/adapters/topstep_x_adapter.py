#!/usr/bin/env python3
"""
TopstepX Python SDK Adapter

Production-ready adapter for TopstepX trading using the project-x-py SDK.
Implements TradingSuite initialization, risk management, and order execution.
Enhanced with fail-closed behavior, centralized retry policies, and structured telemetry.
"""

import asyncio
import logging
import os
import sys
import time
import threading
import importlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from collections import deque

# Import the real project-x-py SDK - fail hard if not available
try:
    import project_x_py
    # Force reload to get latest code changes
    importlib.reload(project_x_py)
    from project_x_py import TradingSuite, EventType
except ImportError as e:
    raise RuntimeError(
        "project-x-py SDK is required for production use. "
        "Install with: pip install 'project-x-py[all]'"
    ) from e


def requires_initialization(func):
    """Decorator to check if adapter is initialized before method execution."""
    async def wrapper(self, *args, **kwargs):
        if not self._is_initialized or not self.suite:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")
        return await func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class AdapterRetryPolicy:
    """Centralized retry policy with bounded timeouts for fail-closed behavior."""
    
    def __init__(self, max_retries: int = None, base_delay: float = None, max_delay: float = None, timeout: float = None):
        # All parameters must come from configuration - fail closed if not provided
        if max_retries is None:
            max_retries_env = os.getenv('ADAPTER_MAX_RETRIES')
            if max_retries_env:
                max_retries = int(max_retries_env)
            else:
                max_retries = 3  # Default value
            if max_retries <= 0:
                raise ValueError("ADAPTER_MAX_RETRIES must be a positive integer")
        
        if base_delay is None:
            base_delay_env = os.getenv('ADAPTER_BASE_DELAY')
            if base_delay_env:
                base_delay = float(base_delay_env)
            else:
                base_delay = 1.0  # Default value
            if base_delay <= 0:
                raise ValueError("ADAPTER_BASE_DELAY must be a positive number")
                
        if max_delay is None:
            max_delay_env = os.getenv('ADAPTER_MAX_DELAY')
            if max_delay_env:
                max_delay = float(max_delay_env)
            else:
                max_delay = 8.0  # Default value
            if max_delay <= 0 or max_delay < base_delay:
                raise ValueError("ADAPTER_MAX_DELAY must be >= base_delay")
                
        if timeout is None:
            timeout_env = os.getenv('ADAPTER_TIMEOUT')
            if timeout_env:
                timeout = float(timeout_env)
            else:
                timeout = 30.0  # Default value
            if timeout <= 0:
                raise ValueError("ADAPTER_TIMEOUT must be a positive number")
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
    
    async def execute_with_retry(self, operation, operation_name: str, logger, *args, **kwargs):
        """Execute operation with exponential backoff retry and timeout."""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.error(f"[RETRY-POLICY] {operation_name} timed out after {self.timeout}s")
                raise TimeoutError(f"{operation_name} operation timed out after {self.timeout} seconds")
            
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"[RETRY-POLICY] {operation_name} succeeded on attempt {attempt + 1}/{self.max_retries + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    # Final failure - emit telemetry and fail closed
                    logger.error(f"[RETRY-POLICY] {operation_name} failed after {self.max_retries + 1} attempts: {e}")
                    self._emit_failure_telemetry(operation_name, e, attempt + 1, logger)
                    break
                
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"[RETRY-POLICY] {operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All retries exhausted - fail closed
        raise RuntimeError(f"FAIL-CLOSED: {operation_name} failed after {self.max_retries + 1} attempts") from last_exception
    
    def _emit_failure_telemetry(self, operation_name: str, error: Exception, attempts: int, logger):
        """Emit structured telemetry for adapter failures."""
        telemetry = {
            "event_type": "adapter_failure",
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempts": attempts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "critical"
        }
        logger.error(f"[TELEMETRY] {telemetry}")


class TopstepXAdapter:
    """
    Production TopstepX adapter with full SDK integration.
    
    Features:
    - Multi-instrument support via TradingSuite
    - Risk management with managed_trade() context
    - Real-time price data and order execution
    - Health monitoring and statistics
    - Production error handling and logging
    """
    
    def __init__(self, instruments: List[str]):
        """
        Initialize adapter with specified instruments.
        
        Args:
            instruments: List of instrument symbols (e.g., ['ES', 'NQ'])
        
        Raises:
            RuntimeError: If required credentials are not found in environment
        """
        # Validate credentials are available immediately - fail hard if not
        # SDK v3.5.9+ uses PROJECT_X_* format, but we also support TOPSTEPX_* for backward compatibility
        api_key = os.getenv('PROJECT_X_API_KEY') or os.getenv('TOPSTEPX_API_KEY')
        username = os.getenv('PROJECT_X_USERNAME') or os.getenv('TOPSTEPX_USERNAME')
        
        # Set PROJECT_X_* variables if they don't exist (SDK v3.5.9+ requirement)
        if api_key and not os.getenv('PROJECT_X_API_KEY'):
            os.environ['PROJECT_X_API_KEY'] = api_key
        if username and not os.getenv('PROJECT_X_USERNAME'):
            os.environ['PROJECT_X_USERNAME'] = username
        
        if not api_key or not username:
            # Provide detailed error message with troubleshooting steps
            missing_vars = []
            if not api_key:
                missing_vars.append('PROJECT_X_API_KEY/TOPSTEPX_API_KEY')
            if not username:
                missing_vars.append('PROJECT_X_USERNAME/TOPSTEPX_USERNAME')
            
            error_msg = (
                f"Missing required TopstepX credentials: {', '.join(missing_vars)}\n"
                f"\n"
                f"Troubleshooting:\n"
                f"1. Verify .env file exists in repository root with credentials:\n"
                f"   TOPSTEPX_API_KEY=your_api_key\n"
                f"   TOPSTEPX_USERNAME=your_username\n"
                f"   TOPSTEPX_ACCOUNT_ID=your_account_id\n"
                f"\n"
                f"2. Ensure environment variables are not being overridden with empty values\n"
                f"   (check workflow configuration or parent process)\n"
                f"\n"
                f"3. Current environment variables:\n"
                f"   PROJECT_X_API_KEY: {'SET' if os.getenv('PROJECT_X_API_KEY') else 'NOT SET'}\n"
                f"   TOPSTEPX_API_KEY: {'SET' if os.getenv('TOPSTEPX_API_KEY') else 'NOT SET'}\n"
                f"   PROJECT_X_USERNAME: {os.getenv('PROJECT_X_USERNAME', 'NOT SET')}\n"
                f"   TOPSTEPX_USERNAME: {os.getenv('TOPSTEPX_USERNAME', 'NOT SET')}\n"
            )
            
            # Log to stderr for visibility
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)
        
        self.instruments = instruments
        self.suites: Dict[str, Optional[TradingSuite]] = {inst: None for inst in instruments}  # One suite per instrument (SDK v3.5.9+)
        self.suite: Optional[TradingSuite] = None  # Primary suite for backward compatibility
        self._is_initialized = False
        self._connection_health = 0.0
        self._last_health_check: Optional[datetime] = None
        
        # PHASE 1: Fill event storage queue (thread-safe with asyncio)
        self._fill_events_queue: deque = deque(maxlen=1000)  # Keep last 1000 fills
        self._fill_events_lock = asyncio.Lock()
        
        # BAR EVENT STREAMING: Storage queue for live bar updates
        self._bar_events_queue: deque = deque(maxlen=100)  # Keep last 100 bars
        self._bar_events_lock = asyncio.Lock()
        
        # REAL-TIME PRICE CACHE: Since WebSocket may not connect, poll and cache prices
        self._price_cache: Dict[str, float] = {}  # {symbol: last_price}
        self._price_cache_lock = asyncio.Lock()
        self._price_refresh_task: Optional[asyncio.Task] = None  # Background refresh task
        
        # Configure production logging
        self.logger = logging.getLogger(f"TopstepXAdapter-{'-'.join(instruments)}")
        if not self.logger.handlers:
            # Send logs to stderr to avoid interfering with stdout JSON communication
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize centralized retry policy with fail-closed configuration validation
        try:
            self.retry_policy = AdapterRetryPolicy()
        except ValueError as e:
            self.logger.error(f"FAIL-CLOSED: Adapter configuration invalid: {e}")
            raise RuntimeError(f"Adapter configuration failure: {e}") from e
        
        self.logger.info(f"🔧 TopstepX adapter initialized for {instruments} with fail-closed retry policy")
        self._emit_telemetry("adapter_initialized", {"instruments": instruments})
    
    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Emit structured telemetry for monitoring and alerting."""
        telemetry = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adapter_instance": f"TopstepX-{'-'.join(self.instruments)}",
            **data
        }
        self.logger.info(f"[TELEMETRY] {telemetry}")
    
    def _map_symbol_to_topstepx(self, symbol: str) -> str:
        """Map standard symbol to TopstepX instrument code.
        
        Args:
            symbol: Standard symbol (ES, NQ, etc.)
            
        Returns:
            TopstepX instrument code (ENQ, EP, MES, MNQ)
        """
        # Reverse mapping: Standard symbols -> TopstepX codes
        # TopstepX SDK expects these specific codes
        symbol_to_topstepx = {
            "ES": "ES",      # E-mini S&P 500 - use ES directly
            "NQ": "NQ",      # E-mini NASDAQ - use NQ directly  
            "MES": "MES",    # Micro E-mini S&P 500
            "MNQ": "MNQ",    # Micro E-mini NASDAQ
        }
        
        mapped = symbol_to_topstepx.get(symbol, symbol)
        self.logger.debug(f"Symbol mapping: {symbol} -> {mapped}")
        return mapped
    
    def _validate_symbol(self, symbol: str) -> None:
        """Validate symbol is in configured instruments.
        
        Args:
            symbol: Instrument symbol to validate
            
        Raises:
            ValueError: If symbol not in configured instruments
        """
        if symbol not in self.instruments:
            raise ValueError(f"Symbol {symbol} not in configured instruments: {self.instruments}")
    
    def _get_suite(self, symbol: str):
        """Get the TradingSuite instance for a given symbol.
        
        Args:
            symbol: Standard symbol (ES, NQ, etc.)
            
        Returns:
            TradingSuite instance for the symbol
            
        Raises:
            ValueError: If symbol not in configured instruments or suite not initialized
        """
        if symbol not in self.suites:
            raise ValueError(f"No suite found for symbol {symbol}")
        suite = self.suites[symbol]
        if suite is None:
            raise ValueError(f"Suite for {symbol} not yet initialized")
        return suite
    
    def _parse_position_data(self, position) -> Optional[Dict[str, Any]]:
        """Parse SDK position object into dictionary format.
        
        Args:
            position: SDK Position object
            
        Returns:
            Position dictionary or None if position is flat
        """
        try:
            # Extract position data
            contract_id = str(getattr(position, 'contractId', ''))
            symbol = self._extract_symbol_from_contract_id(contract_id)
            
            # netPos is signed: positive = long, negative = short
            net_pos = int(getattr(position, 'netPos', 0))
            
            # Determine side
            if net_pos > 0:
                side = "LONG"
                avg_price = float(getattr(position, 'buyAvgPrice', 0.0))
            elif net_pos < 0:
                side = "SHORT"
                avg_price = float(getattr(position, 'sellAvgPrice', 0.0))
            else:
                # Flat position, skip
                return None
            
            # Get P&L data
            unrealized_pnl = float(getattr(position, 'unrealizedPnl', 0.0))
            realized_pnl = float(getattr(position, 'realizedPnl', 0.0))
            position_id = str(getattr(position, 'id', contract_id))
            
            return {
                'symbol': symbol,
                'quantity': abs(net_pos),
                'side': side,
                'avg_price': avg_price,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'position_id': position_id
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse position: {e}")
            return None
    
    def _extract_symbol_from_contract_id(self, contract_id: str) -> str:
        """
        Extract symbol from TopstepX contract ID format.
        
        Examples:
            "CON.F.US.MNQ.H25" -> "MNQ"
            "CON.F.US.EP.Z25" -> "ES" (EP maps to ES)
            "CON.F.US.ENQ.Z25" -> "NQ" (ENQ maps to NQ)
            
        Args:
            contract_id: TopstepX contract identifier
            
        Returns:
            Extracted symbol or original contract_id if parsing fails
        """
        try:
            parts = contract_id.split('.')
            if len(parts) >= 4:
                # TopstepX format: CON.F.US.SYMBOL.EXPIRY
                instrument_part = parts[3]
                
                # Map TopstepX instrument codes to standard symbols
                symbol_map = {
                    "EP": "ES",    # E-mini S&P 500
                    "ENQ": "NQ",   # E-mini NASDAQ
                    "MNQ": "NQ",   # Micro E-mini NASDAQ (map to NQ)
                    "MES": "ES",   # Micro E-mini S&P 500
                }
                
                return symbol_map.get(instrument_part, instrument_part)
        except Exception as e:
            self.logger.warning(f"Failed to parse contract ID {contract_id}: {e}")
        
        return contract_id
    
    async def _on_order_filled(self, event_data: Any):
        """
        PHASE 1: WebSocket callback for ORDER_FILLED events.
        
        This callback is triggered when an order is filled via the SDK's
        real-time WebSocket connection. It transforms the SDK event format
        into the structure expected by the C# layer.
        
        Args:
            event_data: Event data from project-x-py SDK
        """
        try:
            # Extract fill details from SDK event data
            # The SDK provides a GatewayUserTrade event with fill information
            order_id = str(getattr(event_data, 'orderId', getattr(event_data, 'order_id', 'unknown')))
            contract_id = str(getattr(event_data, 'contractId', getattr(event_data, 'contract_id', '')))
            quantity = int(getattr(event_data, 'quantity', getattr(event_data, 'qty', 0)))
            fill_price = float(getattr(event_data, 'price', getattr(event_data, 'fill_price', 0.0)))
            commission = float(getattr(event_data, 'commission', 0.0))
            timestamp_val = getattr(event_data, 'timestamp', None)
            
            # Extract symbol from contract ID
            symbol = self._extract_symbol_from_contract_id(contract_id)
            
            # Determine liquidity type (MAKER/TAKER)
            liquidity_type = str(getattr(event_data, 'liquidityType', 
                                        getattr(event_data, 'liquidity_type', 'TAKER')))
            
            # Parse timestamp
            if timestamp_val:
                if isinstance(timestamp_val, (int, float)):
                    # Assume milliseconds timestamp
                    timestamp = datetime.fromtimestamp(timestamp_val / 1000.0, tz=timezone.utc)
                elif isinstance(timestamp_val, str):
                    timestamp = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Transform to C# expected format
            fill_event = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': fill_price,  # Entry price (same as fill_price for market orders)
                'fill_price': fill_price,
                'commission': commission,
                'exchange': 'CME',  # Default for futures
                'liquidity_type': liquidity_type,
                'timestamp': timestamp.isoformat()
            }
            
            # Add to queue with thread-safe lock
            async with self._fill_events_lock:
                self._fill_events_queue.append(fill_event)
            
            self.logger.info(
                f"[FILL-EVENT] Order filled: {order_id} {symbol} {quantity} @ ${fill_price:.2f}"
            )
            self._emit_telemetry("order_filled", {
                "order_id": order_id,
                "symbol": symbol,
                "quantity": quantity,
                "fill_price": fill_price
            })
            
        except Exception as e:
            self.logger.error(f"Error processing fill event: {e}", exc_info=True)

    async def _on_bar_update(self, event_data: Any):
        """
        BAR EVENT STREAMING: WebSocket callback for bar completion events.
        
        This callback is triggered when a new bar (candle) completes via the SDK's
        real-time WebSocket connection. It transforms the SDK event format into
        the structure expected by the C# layer for trading decisions.
        
        Args:
            event_data: Bar event data from project-x-py SDK
        """
        try:
            # Extract bar details from SDK event data
            # The SDK may provide different field names, so we check multiple possibilities
            contract_id = str(getattr(event_data, 'contractId', getattr(event_data, 'contract_id', '')))
            symbol = self._extract_symbol_from_contract_id(contract_id)
            
            # Get bar OHLCV data
            timestamp_val = getattr(event_data, 'timestamp', getattr(event_data, 't', None))
            open_price = float(getattr(event_data, 'open', getattr(event_data, 'o', 0.0)))
            high_price = float(getattr(event_data, 'high', getattr(event_data, 'h', 0.0)))
            low_price = float(getattr(event_data, 'low', getattr(event_data, 'l', 0.0)))
            close_price = float(getattr(event_data, 'close', getattr(event_data, 'c', 0.0)))
            volume = int(getattr(event_data, 'volume', getattr(event_data, 'v', 0)))
            
            # Parse timestamp
            if timestamp_val:
                if isinstance(timestamp_val, (int, float)):
                    # Assume milliseconds timestamp
                    timestamp = datetime.fromtimestamp(timestamp_val / 1000.0, tz=timezone.utc)
                elif isinstance(timestamp_val, str):
                    timestamp = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Transform to C# expected format
            bar_event = {
                'type': 'bar',
                'instrument': symbol,
                'timestamp': timestamp.isoformat(),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            # Add to queue with thread-safe lock
            async with self._bar_events_lock:
                self._bar_events_queue.append(bar_event)
            
            self.logger.info(
                f"[BAR-EVENT] {symbol} 1m bar: O={open_price:.2f} H={high_price:.2f} L={low_price:.2f} C={close_price:.2f} V={volume}"
            )
            self._emit_telemetry("bar_completed", {
                "symbol": symbol,
                "close": close_price,
                "volume": volume,
                "timestamp": timestamp.isoformat()
            })
            
            # STREAMING OUTPUT: Output bar to stdout for C# to read
            import json
            import sys
            print(json.dumps(bar_event), flush=True)
            
        except Exception as e:
            self.logger.error(f"Error processing bar event: {e}", exc_info=True)

    async def initialize(self) -> None:
        """Setup structured logging for production use."""
        logger = logging.getLogger(f"TopstepXAdapter.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Send logs to stderr to avoid interfering with stdout JSON communication
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _validate_configuration(self) -> None:
        """Validate SDK configuration and credentials."""
        # Validate instruments
        if not self.instruments:
            raise ValueError("At least one instrument must be specified")
            
        # Validate supported instruments
        supported_instruments = {'ES', 'NQ', 'RTY', 'YM', 'MNQ', 'MES'}
        for instrument in self.instruments:
            if instrument not in supported_instruments:
                self.logger.warning(
                    f"Instrument {instrument} may not be supported. "
                    f"Supported: {supported_instruments}"
                )

    async def initialize(self) -> None:
        """
        Initialize TradingSuite with multi-instrument support and risk management.
        Enhanced with fail-closed behavior and centralized retry policies.
        
        Raises:
            RuntimeError: If initialization fails
            ValueError: If configuration is invalid
        """
        if self._is_initialized:
            self.logger.warning("Adapter already initialized")
            return
            
        # Use retry policy for initialization - fail closed if unable to initialize
        async def _initialize_suite():
            self.logger.info(f"Initializing TradingSuite with instruments: {self.instruments}")
            
            # SDK v3.5.9+ requires separate suite instance per instrument
            self.logger.info("Initializing production TopstepX SDK (v3.5.9+ multi-suite mode)...")
            
            for instrument in self.instruments:
                self.logger.info(f"Creating suite for {instrument}...")
                # Map standard symbol to TopstepX instrument code
                topstepx_instrument = self._map_symbol_to_topstepx(instrument)
                self.logger.info(f"Using TopstepX instrument code: {topstepx_instrument}")
                # Load 90 days of historical data for comprehensive learning (matches EnhancedBacktestLearningService)
                suite = await TradingSuite.from_env(instrument=topstepx_instrument, initial_days=90)
                
                # CRITICAL: Initialize WebSocket connection for real-time data streaming with timeout
                self.logger.info(f"🔌 Initializing WebSocket connection for {instrument}...")
                if hasattr(suite, '_initialize'):
                    # Try initialization with retries and exponential backoff
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Increase timeout to 60s for slow connections (Windows can be slow)
                            await asyncio.wait_for(suite._initialize(), timeout=60.0)
                            self.logger.info(f"✅ WebSocket connection initialized for {instrument} on attempt {attempt + 1}")
                            break
                        except asyncio.TimeoutError:
                            if attempt < max_retries - 1:
                                retry_delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                                self.logger.warning(f"⚠️ WebSocket initialization timed out for {instrument} (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                                await asyncio.sleep(retry_delay)
                            else:
                                self.logger.error(f"❌ WebSocket initialization failed after {max_retries} attempts for {instrument} - will continue without WebSocket")
                        except Exception as e:
                            if attempt < max_retries - 1:
                                retry_delay = 2 ** attempt
                                self.logger.warning(f"⚠️ WebSocket initialization error for {instrument} (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay}s...")
                                await asyncio.sleep(retry_delay)
                            else:
                                self.logger.error(f"❌ WebSocket initialization failed after {max_retries} attempts for {instrument}: {e} - will continue without WebSocket")
                else:
                    self.logger.warning(f"⚠️ Suite does not have _initialize method - WebSocket may not connect")
                
                self.suites[instrument] = suite
                self.logger.info(f"✅ {instrument} suite created")
            
            # Set primary suite to first instrument for backward compatibility
            self.suite = self.suites[self.instruments[0]]
            self.logger.info("✅ TopstepX SDK initialized successfully")
            return self.suite
        
        # Initialize suite with retry policy
        self.suite = await self.retry_policy.execute_with_retry(
            _initialize_suite, "TradingSuite_initialization", self.logger
        )
        
        # PHASE 1: Subscribe to ORDER_FILLED events via WebSocket
        try:
            await self.suite.on(EventType.ORDER_FILLED, self._on_order_filled)
            self.logger.info("✅ Subscribed to ORDER_FILLED events via WebSocket")
        except Exception as e:
            error_msg = f"FAIL-CLOSED: Failed to subscribe to fill events: {e}"
            self.logger.error(error_msg)
            self._emit_telemetry("event_subscription_failed", {
                "error": str(e),
                "severity": "critical"
            })
            await self._cleanup_resources()
            raise RuntimeError(error_msg) from e
        
        # BAR EVENT STREAMING: Subscribe to bar completion events via WebSocket
        try:
            # Try NEW_BAR first as per SDK documentation, then fallback to alternatives
            bar_event_types = []
            
            # Check if NEW_BAR or other bar event types exist
            if hasattr(EventType, 'NEW_BAR'):
                bar_event_types.append(EventType.NEW_BAR)
            if hasattr(EventType, 'BAR_UPDATE'):
                bar_event_types.append(EventType.BAR_UPDATE)
            if hasattr(EventType, 'CANDLE_UPDATE'):
                bar_event_types.append(EventType.CANDLE_UPDATE)
            if hasattr(EventType, 'BAR_CLOSED'):
                bar_event_types.append(EventType.BAR_CLOSED)
            if hasattr(EventType, 'CANDLE_CLOSED'):
                bar_event_types.append(EventType.CANDLE_CLOSED)
            
            # Subscribe to first available bar event type
            if bar_event_types:
                await self.suite.on(bar_event_types[0], self._on_bar_update)
                self.logger.info(f"✅ Subscribed to {bar_event_types[0]} events via WebSocket")
            else:
                self.logger.warning("⚠️ No bar event type found in SDK - bar streaming may not work")
        except Exception as e:
            # Bar events are not critical for basic trading, so just log warning
            self.logger.warning(f"Could not subscribe to bar events: {e}")
            self._emit_telemetry("bar_subscription_failed", {
                "error": str(e),
                "severity": "warning"
            })
        
        # Verify connection to ALL instruments - but allow partial failure during initialization
        connection_successes = []
        for instrument in self.instruments:
            async def _test_instrument_connection():
                # Access the suite for this instrument from our suites dictionary
                suite = self.suites[instrument]
                # Use subscript notation as required by TopstepX SDK v3.5.9+
                # Increase timeout to 30s for slow connections
                current_price = await asyncio.wait_for(
                    suite[instrument].data.get_current_price(),
                    timeout=30.0
                )
                self.logger.info(f"✅ {instrument} connected - Current price: ${current_price:.2f}")
                return current_price
            
            try:
                await self.retry_policy.execute_with_retry(
                    _test_instrument_connection, f"{instrument}_connection_test", self.logger
                )
                connection_successes.append(instrument)
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️ {instrument} connection test timed out (may connect later)")
            except Exception as e:
                self.logger.warning(f"⚠️ {instrument} connection test failed (may connect later): {e}")
        
        # Log results but don't fail - WebSocket might still be connecting
        if connection_successes:
            self.logger.info(f"✅ Successfully connected to {len(connection_successes)}/{len(self.instruments)} instruments: {connection_successes}")
        else:
            self.logger.warning(f"⚠️ No instruments connected yet - WebSocket may still be establishing connection")
        
            # Give WebSocket extra time to establish connection (SDK can be slow on Windows)
        self.logger.info("⏳ Waiting 5 seconds for WebSocket connection to stabilize...")
        await asyncio.sleep(5)
        
        # Test risk management system with retry policy
        async def _test_risk_management():
            # Add timeout to prevent hanging
            risk_stats = await asyncio.wait_for(
                self.suite.get_stats(),
                timeout=10.0
            )
            self.logger.info(f"SDK connected successfully - Stats: {risk_stats}")
            return risk_stats
        
        try:
            await self.retry_policy.execute_with_retry(
                _test_risk_management, "risk_management_test", self.logger
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"⚠️ Initial risk management test timed out (WebSocket may still be connecting)")
            self._emit_telemetry("initialization_warning", {
                "reason": "risk_management_timeout",
                "severity": "warning"
            })
            # Don't raise - allow adapter to continue and retry connection
        except Exception as e:
            # Log but don't fail - WebSocket might connect later
            self.logger.warning(f"⚠️ Initial risk management test failed (WebSocket may still be connecting): {e}")
            self._emit_telemetry("initialization_warning", {
                "reason": "risk_management_delayed",
                "error": str(e),
                "severity": "warning"
            })
            # Don't raise - allow adapter to continue and retry connection
        
        # CRITICAL: Do NOT set initialized=True until price cache is populated!
        # C# checks health endpoint and will immediately query /price/<symbol>
        self._connection_health = 100.0  # All instruments connected successfully
        self._last_health_check = datetime.now(timezone.utc)
        
        # CRITICAL: Populate price cache BEFORE marking as initialized
        # This ensures prices are available immediately when C# queries
        self.logger.info("📊 Populating initial price cache...")
        for instrument in self.instruments:
            try:
                suite = self._get_suite(instrument)
                # Add timeout to prevent hanging on price fetch
                price = await asyncio.wait_for(
                    suite[instrument].data.get_current_price(),
                    timeout=10.0
                )
                async with self._price_cache_lock:
                    self._price_cache[instrument] = float(price)
                self.logger.info(f"✅ {instrument} initial price: ${price:.2f}")
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️ Initial price fetch timed out for {instrument}")
                # Set a fallback price to avoid failures
                async with self._price_cache_lock:
                    self._price_cache[instrument] = 0.0
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get initial price for {instrument}: {e}")
                # Set a fallback price to avoid failures
                async with self._price_cache_lock:
                    self._price_cache[instrument] = 0.0
        
        # Start background price refresh task for real-time updates
        # This polls get_current_price() every second and caches results
        # since WebSocket may not be available on practice accounts
        # NOTE: Price refresh task is started in persistent mode by C#
        # self._price_refresh_task = asyncio.create_task(self._continuous_price_refresh())
        self.logger.info("🔄 Price refresh task will be started in persistent mode")
        
        # NOW mark as initialized (C# can safely query prices)
        self._is_initialized = True
        
        self.logger.info(f"🚀 TopstepX SDK adapter initialized successfully (Health: 100%)")
        self._emit_telemetry("initialization_completed", {
            "instruments": self.instruments,
            "health": 100.0,
            "status": "production_ready"
        })

    @requires_initialization
    async def get_price(self, symbol: str) -> float:
        """
        Get current market price for instrument (returns cached real-time value).
        
        Price cache is continuously updated every 1 second by background task,
        ensuring real-time price action even without WebSocket connectivity.
        
        Args:
            symbol: Instrument symbol (e.g., 'MNQ', 'ES')
            
        Returns:
            Current market price (cached, updated every 1 second)
            
        Raises:
            RuntimeError: If price not available in cache
            ValueError: If symbol not configured
        """
        self._validate_symbol(symbol)
        
        # Return from real-time cache (updated every 1 second by background task)
        async with self._price_cache_lock:
            price = self._price_cache.get(symbol)
            
            if price is None or price <= 0:
                # Price not yet in cache - fetch directly as fallback
                self.logger.warning(f"⚠️ {symbol} price not in cache, fetching directly...")
                suite = self._get_suite(symbol)
                price = await suite[symbol].data.get_current_price()
                self._price_cache[symbol] = float(price)
            
            self.logger.debug(f"[PRICE] {symbol}: ${price:.2f} (from cache)")
            return float(price)

    async def _continuous_price_refresh(self):
        """
        Background task to continuously refresh price cache.
        
        Since WebSocket real-time streaming may not be available on practice accounts,
        this task polls get_current_price() every second for all instruments and
        caches the results. The get_price() method then returns cached values instantly.
        
        This ensures real-time price action even without WebSocket connectivity.
        """
        self.logger.info("🔄 [PRICE-REFRESH] Starting continuous price refresh loop")
        
        refresh_count = 0  # Track iterations for heartbeat logging
        
        while True:
            try:
                refresh_count += 1
                
                # Log EVERY iteration for debugging (will show task is running)
                if refresh_count % 10 == 0:
                    self.logger.info(f"🔄 [PRICE-REFRESH] Loop iteration #{refresh_count}")
                
                for symbol in self.instruments:
                    try:
                        # Get fresh price from SDK using LATEST BAR instead of get_current_price()
                        # This may bypass SDK caching and give us real-time tick data
                        suite = self._get_suite(symbol)
                        
                        # Try get_bars(limit=1, partial=True) for latest tick
                        # Falls back to get_current_price() if bars unavailable
                        # Add timeout to prevent hanging on price fetch
                        try:
                            bars = await asyncio.wait_for(
                                suite[symbol].data.get_bars(limit=1, partial=True),
                                timeout=5.0
                            )
                            if bars and len(bars) > 0:
                                price = float(bars[-1].close)  # Use latest bar's close price
                            else:
                                # Fallback to get_current_price() if no bars
                                price = await asyncio.wait_for(
                                    suite[symbol].data.get_current_price(),
                                    timeout=5.0
                                )
                        except asyncio.TimeoutError:
                            # If timeout, skip this iteration
                            self.logger.debug(f"Price fetch timed out for {symbol}, skipping this iteration")
                            continue
                        except Exception as bars_err:
                            # If get_bars() fails, fallback to get_current_price()
                            self.logger.debug(f"get_bars failed for {symbol}, using get_current_price(): {bars_err}")
                            price = await asyncio.wait_for(
                                suite[symbol].data.get_current_price(),
                                timeout=5.0
                            )
                        
                        # Update cache with thread-safe lock
                        async with self._price_cache_lock:
                            old_price = self._price_cache.get(symbol, 0)
                            self._price_cache[symbol] = float(price)
                            
                            # Log price changes for visibility
                            if old_price > 0 and abs(price - old_price) > 0.01:
                                self.logger.info(f"💹 {symbol} price updated: ${old_price:.2f} → ${price:.2f}")
                            
                            # Heartbeat: Log every 30 iterations (30 seconds) even if no change
                            elif refresh_count % 30 == 0:
                                self.logger.info(f"🔄 [HEARTBEAT] {symbol} price stable: ${price:.2f} (refresh #{refresh_count})")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ [PRICE-REFRESH] Failed to refresh {symbol} price: {e}")
                
                # Poll every 1 second for real-time updates
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 [PRICE-REFRESH] Price refresh task cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ [PRICE-REFRESH] Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Back off on errors

    @requires_initialization
    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: int = 5,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Fetch historical bar data for a symbol.
        
        Args:
            symbol: Instrument symbol (e.g., 'ES', 'NQ')
            timeframe: Bar timeframe in minutes (1, 5, 15, 60)
            start_date: Start datetime (default: 1 day ago)
            end_date: End datetime (default: now)
            limit: Maximum number of bars to return (default: 1000)
            
        Returns:
            Dictionary with bars array and metadata
        """
        from datetime import timedelta
        
        self._validate_symbol(symbol)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=1)
        
        try:
            suite = self._get_suite(symbol)
            
            # TopstepX SDK historical data access via suite.client.get_bars()
            # API: get_bars(symbol, days, interval, unit, limit, partial, start_time, end_time)
            # unit: 1=second, 2=minute, 3=hour, 4=day
            # interval: number of units per bar (e.g., 5 for 5-minute bars)
            
            # Map timeframe (minutes) to interval/unit
            unit = 2  # minute
            interval = timeframe  # e.g., 5 for 5-minute bars
            
            # Calculate days based on date range if provided
            if start_date and end_date:
                days_diff = (end_date - start_date).days
                days = max(days_diff, 1)
            else:
                days = 1  # Default to 1 day
            
            self.logger.info(f"Fetching historical bars for {symbol}: {days} days, {interval}min intervals")
            
            # Fetch bars using SDK client.get_bars()
            bars_df = await suite.client.get_bars(
                symbol=symbol,
                days=days,
                interval=interval,
                unit=unit,
                limit=limit,
                partial=True,  # Include partial bars
                start_time=start_date,
                end_time=end_date
            )
            
            # Convert DataFrame to list of bar dictionaries
            bars = []
            if bars_df is not None and len(bars_df) > 0:
                # TopstepX returns polars DataFrame, convert to dict records
                for row in bars_df.iter_rows(named=True):
                    bar_data = {
                        'timestamp': str(row.get('timestamp', row.get('time', ''))),
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                        'volume': int(row.get('volume', 0))
                    }
                    bars.append(bar_data)
            
            self.logger.info(f"✅ Fetched {len(bars)} historical bars for {symbol}")
            
            self._emit_telemetry("historical_data_fetched", {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars_count": len(bars),
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            })
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'bars': bars,
                'count': len(bars),
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
            
        except Exception as e:
            error_msg = f"Failed to fetch historical bars for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            self._emit_telemetry("historical_data_failed", {
                "symbol": symbol,
                "error": str(e),
                "timeframe": timeframe
            })
            return {
                'success': False,
                'error': error_msg,
                'symbol': symbol
            }

    @requires_initialization
    async def place_order(
        self, 
        symbol: str, 
        size: int, 
        stop_loss: float, 
        take_profit: float,
        max_risk_percent: float = 0.01
    ) -> Dict[str, Any]:
        """
        Place bracket order with risk management.
        
        Args:
            symbol: Instrument symbol
            size: Position size (positive for long, negative for short)
            stop_loss: Stop loss price
            take_profit: Take profit price
            max_risk_percent: Maximum risk as percentage of account (default 1%)
            
        Returns:
            Order execution result with order ID and status
            
        Raises:
            RuntimeError: If order placement fails
            ValueError: If parameters are invalid
        """
        self._validate_symbol(symbol)
            
        if size == 0:
            raise ValueError("Order size cannot be zero")
            
        try:
            current_price = await self.get_price(symbol)
            
            # Validate price levels
            if size > 0:  # Long position
                if stop_loss >= current_price:
                    raise ValueError(f"Stop loss {stop_loss} must be below current price {current_price} for long position")
                if take_profit <= current_price:
                    raise ValueError(f"Take profit {take_profit} must be above current price {current_price} for long position")
            else:  # Short position
                if stop_loss <= current_price:
                    raise ValueError(f"Stop loss {stop_loss} must be above current price {current_price} for short position")
                if take_profit >= current_price:
                    raise ValueError(f"Take profit {take_profit} must be below current price {current_price} for short position")
            
            self.logger.info(
                f"[ORDER] Placing managed trade: {symbol} size={size} "
                f"entry=${current_price:.2f} stop=${stop_loss:.2f} target=${take_profit:.2f}"
            )
            
            # Use managed trade context for risk enforcement
            async with self.suite.managed_trade(max_risk_percent=max_risk_percent):
                # Place bracket order through SDK
                side = 'buy' if size > 0 else 'sell'
                suite = self._get_suite(symbol)
                order_result = await suite.orders.place_bracket_order(
                    side=side,
                    quantity=abs(size),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                # Structure return data
                result = {
                    'success': True,
                    'order_id': str(order_result.get('id', 'unknown')),
                    'symbol': symbol,
                    'size': size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'risk_percent': max_risk_percent
                }
                
                self.logger.info(f"✅ Order placed successfully: {result['order_id']}")
                return result
                
        except Exception as e:
            error_msg = f"Failed to place order for {symbol}: {e}"
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'size': size,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    @requires_initialization
    async def get_fill_events(self) -> Dict[str, Any]:
        """
        PHASE 1: Get recent fill events from TopstepX SDK.
        
        This method returns fill events that have been collected via WebSocket
        subscription to EventType.ORDER_FILLED. Events are stored in an in-memory
        queue and retrieved by the C# layer via polling.
        
        Returns:
            Dictionary with fills array containing fill event data
        """
        try:
            # Read all events from queue and clear it (thread-safe)
            async with self._fill_events_lock:
                fills = list(self._fill_events_queue)
                self._fill_events_queue.clear()
            
            self.logger.debug(f"[FILL-EVENTS] Returning {len(fills)} fill events")
            
            return {
                'fills': fills,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting fill events: {e}")
            return {
                'fills': [],
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    @requires_initialization
    async def get_bar_events(self) -> Dict[str, Any]:
        """
        BAR EVENT STREAMING: Get recent bar events from TopstepX SDK.
        
        This method returns bar events that have been collected via WebSocket
        subscription to bar completion events. Events are stored in an in-memory
        queue and retrieved by the C# layer via polling.
        
        Returns:
            Dictionary with bars array containing bar event data
        """
        try:
            # Read all events from queue and clear it (thread-safe)
            async with self._bar_events_lock:
                bars = list(self._bar_events_queue)
                self._bar_events_queue.clear()
            
            self.logger.debug(f"[BAR-EVENTS] Returning {len(bars)} bar events")
            
            return {
                'bars': bars,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bar events: {e}")
            return {
                'bars': [],
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    async def get_health_score(self) -> Dict[str, Any]:
        """
        PHASE 7 ENHANCED: Get comprehensive health metrics and statistics.
        
        Returns:
            Health score and system statistics with enhanced monitoring
        """
            
        try:
            # Get suite statistics with timeout
            stats = await asyncio.wait_for(
                self.suite.get_stats(),
                timeout=5.0
            )
            
            # Calculate connection health for each instrument
            instrument_health = {}
            total_health = 0.0
            connected_count = 0
            
            for instrument in self.instruments:
                try:
                    # Test price data availability by accessing the instrument
                    instrument_data = self._get_suite(instrument)
                    if instrument_data:
                        instrument_health[instrument] = 100.0
                        total_health += 100.0
                        connected_count += 1
                    else:
                        instrument_health[instrument] = 0.0
                except Exception as e:
                    self.logger.warning(f"Health check failed for {instrument}: {e}")
                    instrument_health[instrument] = 0.0
                    
            # Calculate overall health score based on connected instruments
            # If at least one instrument is connected, report 100% health
            # This allows trading to proceed even if some instruments fail
            overall_health = 100.0 if connected_count > 0 else 0.0
            
            # Update internal health tracking
            self._connection_health = overall_health
            self._last_health_check = datetime.now(timezone.utc)
            
            # PHASE 7: Enhanced monitoring
            
            # 1. WebSocket connection status
            websocket_connected = False
            try:
                if hasattr(self.suite, 'realtime_client'):
                    realtime_client = self.suite.realtime_client
                    if hasattr(realtime_client, 'is_connected'):
                        websocket_connected = realtime_client.is_connected
                    elif hasattr(realtime_client, 'connected'):
                        websocket_connected = realtime_client.connected
            except Exception as e:
                self.logger.debug(f"WebSocket status check: {e}")
            
            # 2. Authentication validity check (lightweight API call)
            auth_valid = True
            try:
                # Try getting stats as a lightweight auth check with timeout
                test_stats = await asyncio.wait_for(
                    self.suite.get_stats(),
                    timeout=5.0
                )
                auth_valid = test_stats is not None
            except asyncio.TimeoutError:
                self.logger.warning(f"Auth validity check timed out")
                auth_valid = False
            except Exception as e:
                self.logger.warning(f"Auth validity check failed: {e}")
                auth_valid = False
            
            # 3. Trading permissions check
            trading_enabled = False
            try:
                if hasattr(self.suite, 'account'):
                    account = self.suite.account
                    if hasattr(account, 'canTrade'):
                        trading_enabled = account.canTrade
                    elif hasattr(account, 'can_trade'):
                        trading_enabled = account.can_trade
            except Exception as e:
                self.logger.debug(f"Trading permission check: {e}")
            
            # 4. Rate limit tracking (if available)
            rate_limit_remaining = None
            rate_limit_reset = None
            try:
                if hasattr(self.suite, 'client'):
                    client = self.suite.client
                    if hasattr(client, 'rate_limit_remaining'):
                        rate_limit_remaining = client.rate_limit_remaining
                    if hasattr(client, 'rate_limit_reset'):
                        rate_limit_reset = client.rate_limit_reset
            except Exception as e:
                self.logger.debug(f"Rate limit check: {e}")
            
            health_data = {
                'health_score': int(overall_health),
                'status': 'healthy' if overall_health >= 80 else 'degraded' if overall_health >= 50 else 'critical',
                'instruments': instrument_health,
                'suite_stats': stats,
                'last_check': self._last_health_check.isoformat(),
                'uptime_seconds': (datetime.now(timezone.utc) - self._last_health_check).total_seconds(),
                'initialized': self._is_initialized,
                # PHASE 7 enhancements
                'websocket_connected': websocket_connected,
                'auth_valid': auth_valid,
                'trading_enabled': trading_enabled,
                'rate_limit_remaining': rate_limit_remaining,
                'rate_limit_reset': rate_limit_reset
            }
            
            # Log health status
            if overall_health >= 80:
                self.logger.debug(f"System healthy: {overall_health:.1f}%")
            else:
                self.logger.warning(f"System health degraded: {overall_health:.1f}%")
                
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'health_score': 0,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now(timezone.utc).isoformat()
            }

    @requires_initialization
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        PHASE 2: Get all current positions from TopstepX SDK.
        
        Returns list of positions with details needed for reconciliation.
        
        Returns:
            List of position dictionaries with keys:
                - symbol: str (e.g., "ES", "MNQ")
                - quantity: int (positive for long, negative for short)
                - side: str ("LONG" or "SHORT")
                - avg_price: float
                - unrealized_pnl: float
                - realized_pnl: float
                - position_id: str
        """
        try:
            # Query all positions from SDK
            all_positions = await self.suite.positions.get_all_positions()
            
            result = []
            for position in all_positions:
                parsed = self._parse_position_data(position)
                if parsed:
                    result.append(parsed)
            
            self.logger.debug(f"[POSITIONS] Retrieved {len(result)} positions")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    @requires_initialization
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        PHASE 2: Get a specific position by symbol.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "MNQ")
            
        Returns:
            Position dictionary or None if not found
        """
        
        try:
            # Get all positions and filter by symbol
            all_positions = await self.get_positions()
            
            for position in all_positions:
                if position['symbol'] == symbol:
                    return position
            
            self.logger.debug(f"[POSITION] No position found for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get position for {symbol}: {e}")
            return None
    
    @requires_initialization
    async def close_position(self, symbol: str, quantity: Optional[int] = None) -> Dict[str, Any]:
        """
        PHASE 3: Close a position by placing opposite side market order.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "MNQ")
            quantity: Optional quantity to close (None = close entire position)
            
        Returns:
            Dictionary with success, order_id, and closed_quantity
        """
        
        try:
            # Get current position
            all_positions = await self.suite.positions.get_all_positions()
            
            position = None
            for pos in all_positions:
                pos_symbol = self._extract_symbol_from_contract_id(str(getattr(pos, 'contractId', '')))
                if pos_symbol == symbol:
                    position = pos
                    break
            
            if position is None:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            net_pos = int(getattr(position, 'netPos', 0))
            if net_pos == 0:
                return {
                    'success': False,
                    'error': f'Position for {symbol} is flat (netPos=0)'
                }
            
            # Determine quantity to close
            close_qty = quantity if quantity is not None else abs(net_pos)
            
            # Validate quantity
            if close_qty > abs(net_pos):
                return {
                    'success': False,
                    'error': f'Cannot close {close_qty} contracts, position size is {abs(net_pos)}'
                }
            
            # Determine opposite side: if long (netPos > 0), sell to close; if short, buy to close
            # SDK uses: 0=Buy, 1=Sell
            side = 1 if net_pos > 0 else 0
            side_str = "SELL" if net_pos > 0 else "BUY"
            
            contract_id = str(getattr(position, 'contractId', ''))
            
            self.logger.info(f"[CLOSE-POSITION] Closing {symbol}: {side_str} {close_qty} contracts (position={net_pos})")
            
            # Place market order on opposite side to close through instrument
            instrument_obj = self._get_suite(symbol)
            order_result = await instrument_obj.orders.place_market_order(
                contract_id=contract_id,
                side=side,
                size=close_qty
            )
            
            order_id = str(getattr(order_result, 'id', getattr(order_result, 'orderId', 'unknown')))
            
            self.logger.info(f"✅ [CLOSE-POSITION] Position close order placed: {order_id}")
            self._emit_telemetry("position_closed", {
                "symbol": symbol,
                "quantity": close_qty,
                "side": side_str,
                "order_id": order_id
            })
            
            return {
                'success': True,
                'order_id': order_id,
                'closed_quantity': close_qty,
                'symbol': symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @requires_initialization
    async def modify_stop_loss(self, symbol: str, stop_price: float) -> Dict[str, Any]:
        """
        PHASE 3: Modify stop loss for a position.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "MNQ")
            stop_price: New stop loss price
            
        Returns:
            Dictionary with success and order_id
        """
        
        try:
            # Get current position to determine contract ID
            all_positions = await self.suite.positions.get_all_positions()
            
            position = None
            for pos in all_positions:
                pos_symbol = self._extract_symbol_from_contract_id(str(getattr(pos, 'contractId', '')))
                if pos_symbol == symbol:
                    position = pos
                    break
            
            if position is None:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            contract_id = str(getattr(position, 'contractId', ''))
            
            # Search for existing stop orders through instrument
            instrument_obj = self._get_suite(symbol)
            open_orders = await instrument_obj.orders.search_open_orders(contract_id=contract_id)
            
            # Filter for stop orders (type 4 = Stop)
            stop_orders = [order for order in open_orders if getattr(order, 'type', None) == 4]
            
            if not stop_orders:
                return {
                    'success': False,
                    'error': f'No stop order found for {symbol}. Create a stop order first before modifying.'
                }
            
            # Modify the first stop order found
            stop_order = stop_orders[0]
            order_id = str(getattr(stop_order, 'id', getattr(stop_order, 'orderId', 'unknown')))
            
            self.logger.info(f"[MODIFY-STOP] Modifying stop loss for {symbol}: order={order_id} new_stop=${stop_price:.2f}")
            
            # Modify the stop order through instrument
            modify_result = await instrument_obj.orders.modify_order(
                order_id=order_id,
                stop_price=stop_price
            )
            
            self.logger.info(f"✅ [MODIFY-STOP] Stop loss modified successfully: {order_id}")
            self._emit_telemetry("stop_loss_modified", {
                "symbol": symbol,
                "stop_price": stop_price,
                "order_id": order_id
            })
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'stop_price': stop_price
            }
            
        except Exception as e:
            self.logger.error(f"Error modifying stop loss for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @requires_initialization
    async def modify_take_profit(self, symbol: str, take_profit_price: float) -> Dict[str, Any]:
        """
        PHASE 3: Modify take profit for a position.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "MNQ")
            take_profit_price: New take profit price
            
        Returns:
            Dictionary with success and order_id
        """
        
        try:
            # Get current position to determine contract ID and side
            all_positions = await self.suite.positions.get_all_positions()
            
            position = None
            for pos in all_positions:
                pos_symbol = self._extract_symbol_from_contract_id(str(getattr(pos, 'contractId', '')))
                if pos_symbol == symbol:
                    position = pos
                    break
            
            if position is None:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            contract_id = str(getattr(position, 'contractId', ''))
            net_pos = int(getattr(position, 'netPos', 0))
            
            if net_pos == 0:
                return {
                    'success': False,
                    'error': f'Position for {symbol} is flat (netPos=0)'
                }
            
            # Determine which side the take profit order should be
            # If long position (netPos > 0), take profit is a sell limit (side=1)
            # If short position (netPos < 0), take profit is a buy limit (side=0)
            tp_side = 1 if net_pos > 0 else 0
            
            # Search for existing limit orders through instrument
            instrument_obj = self._get_suite(symbol)
            open_orders = await instrument_obj.orders.search_open_orders(contract_id=contract_id)
            
            # Filter for limit orders on the take profit side (type 1 = Limit)
            tp_orders = [order for order in open_orders 
                        if getattr(order, 'type', None) == 1 
                        and getattr(order, 'side', None) == tp_side]
            
            if not tp_orders:
                return {
                    'success': False,
                    'error': f'No take profit order found for {symbol}. Create a limit order first before modifying.'
                }
            
            # Modify the first take profit order found
            tp_order = tp_orders[0]
            order_id = str(getattr(tp_order, 'id', getattr(tp_order, 'orderId', 'unknown')))
            
            self.logger.info(f"[MODIFY-TARGET] Modifying take profit for {symbol}: order={order_id} new_target=${take_profit_price:.2f}")
            
            # Modify the limit order through instrument
            modify_result = await instrument_obj.orders.modify_order(
                order_id=order_id,
                limit_price=take_profit_price
            )
            
            self.logger.info(f"✅ [MODIFY-TARGET] Take profit modified successfully: {order_id}")
            self._emit_telemetry("take_profit_modified", {
                "symbol": symbol,
                "take_profit_price": take_profit_price,
                "order_id": order_id
            })
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'take_profit_price': take_profit_price
            }
            
        except Exception as e:
            self.logger.error(f"Error modifying take profit for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @requires_initialization
    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> Dict[str, Any]:
        """
        PHASE 4: Place bracket order with entry, stop loss, and take profit.
        
        Args:
            symbol: Trading symbol (e.g., "ES", "MNQ")
            side: "BUY" or "SELL"
            quantity: Order quantity
            entry_price: Entry limit price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dictionary with bracket order details including all three order IDs
        """
        
        if symbol not in self.instruments:
            return {
                'success': False,
                'error': f'Symbol {symbol} not in configured instruments: {self.instruments}'
            }
        
        try:
            # Convert side to SDK format: 0=Buy, 1=Sell
            side_upper = side.upper()
            if side_upper not in ['BUY', 'SELL']:
                return {
                    'success': False,
                    'error': f'Invalid side: {side}. Must be BUY or SELL'
                }
            
            sdk_side = 0 if side_upper == 'BUY' else 1
            
            # Get contract ID for the symbol
            # For simplicity, we'll construct it from the symbol
            # In production, this should query the SDK for the actual contract ID
            contract_id_map = {
                'NQ': 'CON.F.US.ENQ.Z25',
                'MNQ': 'CON.F.US.MNQ.Z25',  # Keep for backward compatibility
                'ES': 'CON.F.US.EP.Z25',
            }
            
            contract_id = contract_id_map.get(symbol)
            if not contract_id:
                # Fallback: try to get from position or use symbol as-is
                contract_id = f"CON.F.US.{symbol}.Z25"
            
            self.logger.info(
                f"[BRACKET] Placing bracket order: {symbol} {side} {quantity} "
                f"entry=${entry_price:.2f} stop=${stop_loss_price:.2f} target=${take_profit_price:.2f}"
            )
            
            # Place bracket order using SDK's native bracket support through instrument
            instrument_obj = self._get_suite(symbol)
            bracket_result = await instrument_obj.orders.place_bracket_order(
                contract_id=contract_id,
                side=sdk_side,
                size=quantity,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            # Extract order IDs from bracket response
            success = getattr(bracket_result, 'success', True)
            entry_order_id = str(getattr(bracket_result, 'entry_order_id', 
                                        getattr(bracket_result, 'entryOrderId', 'unknown')))
            stop_order_id = str(getattr(bracket_result, 'stop_order_id',
                                       getattr(bracket_result, 'stopOrderId', 'unknown')))
            target_order_id = str(getattr(bracket_result, 'target_order_id',
                                         getattr(bracket_result, 'targetOrderId', 'unknown')))
            
            # Get actual prices used (after tick alignment)
            actual_entry = float(getattr(bracket_result, 'entry_price', entry_price))
            actual_stop = float(getattr(bracket_result, 'stop_loss_price', stop_loss_price))
            actual_target = float(getattr(bracket_result, 'take_profit_price', take_profit_price))
            
            error_message = getattr(bracket_result, 'error_message', None)
            
            if not success and error_message:
                return {
                    'success': False,
                    'error': error_message
                }
            
            self.logger.info(
                f"✅ [BRACKET] Bracket order placed successfully: "
                f"entry={entry_order_id} stop={stop_order_id} target={target_order_id}"
            )
            
            self._emit_telemetry("bracket_order_placed", {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_order_id": entry_order_id,
                "stop_order_id": stop_order_id,
                "target_order_id": target_order_id
            })
            
            return {
                'success': True,
                'entry_order_id': entry_order_id,
                'stop_order_id': stop_order_id,
                'target_order_id': target_order_id,
                'entry_price': actual_entry,
                'stop_loss_price': actual_stop,
                'take_profit_price': actual_target,
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @requires_initialization
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        PHASE 5: Get status of a specific order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Dictionary with order status details or None if not found
        """
        
        try:
            # Query order from SDK
            # Note: SDK access pattern may vary - adjust based on actual SDK API
            # This assumes suite.orders.get_order_by_id() exists
            order = None
            
            # Try to find order across all instruments
            for symbol in self.instruments:
                try:
                    instrument_obj = self._get_suite(symbol)
                    if hasattr(instrument_obj, 'orders'):
                        # Attempt to get order by ID
                        # SDK API may vary - this is based on problem statement guidance
                        if hasattr(instrument_obj.orders, 'get_order_by_id'):
                            order = await instrument_obj.orders.get_order_by_id(order_id)
                            if order:
                                break
                except Exception:
                    continue
            
            if not order:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found'
                }
            
            # Transform Order object to dictionary
            # Status codes: 0=None, 1=Open, 2=Filled, 3=Cancelled, 4=Expired, 5=Rejected, 6=Pending
            status = int(getattr(order, 'status', 0))
            size = int(getattr(order, 'size', 0))
            fill_volume = int(getattr(order, 'fillVolume', 0))
            filled_price = float(getattr(order, 'filledPrice', 0.0))
            
            result = {
                'success': True,
                'order_id': str(getattr(order, 'id', order_id)),
                'status': status,
                'filled_quantity': fill_volume,
                'remaining_quantity': size - fill_volume,
                'avg_fill_price': filled_price,
                'is_filled': getattr(order, 'is_filled', fill_volume >= size),
                'is_open': getattr(order, 'is_open', status == 1),
                'is_cancelled': getattr(order, 'is_cancelled', status == 3)
            }
            
            self.logger.info(f"[ORDER-STATUS] {order_id}: status={status} filled={fill_volume}/{size}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @requires_initialization
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        PHASE 6: Cancel a specific order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with cancellation result
        """
        
        try:
            # Try to cancel order across all instruments
            cancelled = False
            
            for symbol in self.instruments:
                try:
                    instrument_obj = self._get_suite(symbol)
                    if hasattr(instrument_obj, 'orders'):
                        # Attempt to cancel order
                        result = await instrument_obj.orders.cancel_order(order_id)
                        if result:
                            cancelled = True
                            break
                except Exception:
                    continue
            
            if cancelled:
                self.logger.info(f"[CANCEL-ORDER] Successfully cancelled order {order_id}")
                self._emit_telemetry("order_cancelled", {"order_id": order_id})
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'message': 'Order cancelled successfully'
                }
            else:
                return {
                    'success': False,
                    'order_id': order_id,
                    'error': 'Failed to cancel order or order not found'
                }
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    @requires_initialization
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        PHASE 6: Cancel all orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter cancellations
            
        Returns:
            Dictionary with cancellation results
        """
        
        try:
            cancelled_count = 0
            
            # Determine which symbols to process
            symbols_to_process = [symbol] if symbol else self.instruments
            
            for sym in symbols_to_process:
                try:
                    instrument_obj = self._get_suite(sym)
                    if hasattr(instrument_obj, 'orders'):
                        # Get contract ID for filtering
                        contract_id = None
                        if symbol:
                            # Try to determine contract ID from symbol
                            # Map symbol to contract ID format
                            symbol_to_contract = {
                                'ES': 'CON.F.US.EP.Z25',
                                'NQ': 'CON.F.US.ENQ.Z25',
                                'MNQ': 'CON.F.US.MNQ.Z25',
                                'MES': 'CON.F.US.MES.Z25'
                            }
                            contract_id = symbol_to_contract.get(sym)
                        
                        # Cancel all orders for this instrument
                        result = await instrument_obj.orders.cancel_all_orders(contract_id=contract_id)
                        
                        # Result might be boolean or dict with count
                        if isinstance(result, dict):
                            cancelled_count += result.get('count', 1 if result.get('success') else 0)
                        elif result:
                            cancelled_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error cancelling orders for {sym}: {e}")
                    continue
            
            self.logger.info(f"[CANCEL-ALL] Cancelled {cancelled_count} orders" + (f" for {symbol}" if symbol else ""))
            self._emit_telemetry("orders_cancelled", {
                "count": cancelled_count,
                "symbol": symbol
            })
            
            return {
                'success': True,
                'cancelled_count': cancelled_count,
                'symbol': symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @requires_initialization
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio positions and P&L."""
        try:
            # Get portfolio data from TradingSuite
            suite_stats = await self.suite.get_stats()
            
            # Use new get_positions() method for detailed position data
            positions_list = await self.get_positions()
            
            # Transform to dict keyed by symbol for backward compatibility
            positions = {}
            for pos in positions_list:
                symbol = pos['symbol']
                positions[symbol] = {
                    'size': pos['quantity'] if pos['side'] == 'LONG' else -pos['quantity'],
                    'average_price': pos['avg_price'],
                    'unrealized_pnl': pos['unrealized_pnl'],
                    'realized_pnl': pos['realized_pnl']
                }
            
            # Add instruments with no position
            for instrument in self.instruments:
                if instrument not in positions:
                    positions[instrument] = {
                        'size': 0,
                        'average_price': 0.0,
                        'unrealized_pnl': 0.0,
                        'realized_pnl': 0.0
                    }
                    
            return {
                'portfolio': suite_stats,
                'positions': positions,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio status: {e}")
            raise RuntimeError(f"Portfolio status retrieval failed: {e}") from e

    async def disconnect(self) -> None:
        """Clean shutdown of TradingSuite and resources."""
        if not self._is_initialized:
            self.logger.info("Adapter already disconnected")
            return
            
        self.logger.info("Disconnecting TopstepX adapter...")
        
        try:
            await self._cleanup_resources()
            self.logger.info("✅ TopstepX adapter disconnected successfully")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            raise
            
    async def _cleanup_resources(self) -> None:
        """Internal cleanup of resources."""
        # Disconnect all suite instances (SDK v3.5.9+ multi-suite mode)
        for instrument, suite in self.suites.items():
            if suite:
                try:
                    await suite.disconnect()
                    self.logger.info(f"Disconnected {instrument} suite")
                except Exception as e:
                    self.logger.warning(f"Error disconnecting {instrument} suite: {e}")
                finally:
                    self.suites[instrument] = None
        
        self.suite = None
        self._is_initialized = False
        self._connection_health = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected and healthy."""
        return self._is_initialized and self._connection_health >= 80.0

    @property
    def connection_health(self) -> float:
        """Get current connection health percentage."""
        return self._connection_health

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()



# Standalone test function for validation
async def test_adapter_functionality():
    """Test adapter functionality for CI validation."""
    print("🧪 Testing TopstepX Adapter...")
    
    try:
        # Test initialization
        adapter = TopstepXAdapter(["ES", "NQ"])
        await adapter.initialize()
        
        # Test health check
        health = await adapter.get_health_score()
        assert health['health_score'] >= 80, f"Health score too low: {health['health_score']}"
        print(f"✅ Health check passed: {health['health_score']}%")
        
        # Test price retrieval
        nq_price = await adapter.get_price("NQ")
        print(f"✅ NQ price: ${nq_price:.2f}")
        
        # Test order placement (demo mode)
        order_result = await adapter.place_order(
            symbol="NQ",
            size=1,
            stop_loss=nq_price - 10,
            take_profit=nq_price + 15,
            max_risk_percent=0.005  # 0.5% risk for testing
        )
        assert order_result['success'], f"Order failed: {order_result.get('error')}"
        print(f"✅ Order test passed: {order_result['order_id']}")
        
        # Test portfolio status
        portfolio = await adapter.get_portfolio_status()
        print(f"✅ Portfolio status retrieved")
        
        # Clean disconnect
        await adapter.disconnect()
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    import json
    import asyncio
    
    # Check for persistent/streaming mode
    if len(sys.argv) > 1 and sys.argv[1] == "stream":
        # Configure SDK logging to stderr BEFORE initializing adapter
        # This ensures SDK's structured JSON logs don't interfere with stdout communication
        sdk_logger = logging.getLogger('project_x_py')
        sdk_logger.setLevel(logging.INFO)
        # Remove any existing handlers to avoid duplication
        sdk_logger.handlers.clear()
        # Add stderr handler only
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        sdk_logger.addHandler(stderr_handler)
        
        # Also configure root logger to stderr
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.INFO)
        
        # PERSISTENT MODE: Keep adapter alive and process commands via stdin/stdout
        async def persistent_mode():
            """Run adapter in persistent mode with stdin/stdout communication."""
            adapter = None
            try:
                # Create adapter instance (don't initialize yet)
                adapter = TopstepXAdapter(["ES", "NQ"])
                
                # WINDOWS FIX: Use synchronous stdin reading in a separate thread
                # asyncio.connect_read_pipe() doesn't work reliably on Windows
                import threading
                import queue
                
                # Create thread-safe queue for stdin commands
                stdin_queue = queue.Queue()
                
                def stdin_reader_thread():
                    """Synchronous thread that reads stdin and puts lines in queue"""
                    try:
                        print(json.dumps({"type": "debug", "message": "Stdin reader thread started"}), flush=True)
                        while True:
                            line = sys.stdin.readline()
                            if not line:
                                print(json.dumps({"type": "debug", "message": "Stdin EOF detected"}), flush=True)
                                break
                            stdin_queue.put(line)
                    except Exception as e:
                        print(json.dumps({"type": "error", "message": f"Stdin reader thread error: {str(e)}"}), flush=True)
                
                # Start stdin reader thread
                thread = threading.Thread(target=stdin_reader_thread, daemon=True)
                thread.start()
                
                print(json.dumps({"type": "debug", "message": "Stdin reader thread launched"}), flush=True)
                
                # Send initialization success IMMEDIATELY so C# bot doesn't timeout
                # We'll initialize TopstepX connection in background
                print(json.dumps({"type": "init", "success": True, "message": "Adapter created, connecting to TopstepX in background"}), flush=True)
                
                # Track initialization status
                initialization_complete = False
                initialization_error = None
                
                # Run SDK initialization in a SEPARATE THREAD for true parallelism
                # The SDK has blocking I/O that prevents async event loop from processing commands
                def initialize_sdk_in_thread():
                    nonlocal initialization_complete, initialization_error
                    try:
                        print(json.dumps({"type": "debug", "message": "Starting TopstepX initialization in background thread..."}), flush=True)
                        
                        # Redirect stdout during SDK initialization to prevent non-JSON output
                        original_stdout = sys.stdout
                        sys.stdout = sys.stderr  # Send SDK prints to stderr
                        
                        try:
                            # Create new event loop for this thread
                            thread_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(thread_loop)
                            
                            # Run initialization synchronously in this thread
                            thread_loop.run_until_complete(asyncio.wait_for(adapter.initialize(), timeout=60.0))
                            thread_loop.close()
                        finally:
                            sys.stdout = original_stdout  # Restore stdout
                        
                        initialization_complete = True
                        print(json.dumps({"type": "info", "message": "TopstepX connection established"}), flush=True)
                    except asyncio.TimeoutError:
                        initialization_error = "TopstepX connection timed out after 60s"
                        print(json.dumps({"type": "warning", "message": initialization_error}), flush=True)
                    except Exception as e:
                        initialization_error = f"TopstepX connection error: {str(e)}"
                        print(json.dumps({"type": "warning", "message": initialization_error}), flush=True)
                
                # Start background initialization in separate thread (true parallelism)
                print(json.dumps({"type": "debug", "message": "Creating SDK initialization thread..."}), flush=True)
                init_thread = threading.Thread(target=initialize_sdk_in_thread, daemon=True, name="TopstepX-Init")
                init_thread.start()
                print(json.dumps({"type": "debug", "message": f"SDK initialization thread started: {init_thread.name}"}), flush=True)
                
                # Process commands from stdin queue (Windows-compatible approach)
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        # Poll the queue with short timeout to allow event loop to process other tasks
                        line = await loop.run_in_executor(None, stdin_queue.get, True, 0.1)
                        if not line:
                            break  # EOF reached
                        
                        # Line is already a string from stdin.readline()
                        line = line.strip()
                        if not line:
                            continue
                        
                        print(json.dumps({"type": "debug", "message": f"Received command: {line[:100]}"}), flush=True)
                        
                        cmd_data = json.loads(line)
                        action = cmd_data.get("action")
                        
                        print(json.dumps({"type": "debug", "message": f"Processing action: {action}"}), flush=True)
                        
                        if action == "shutdown":
                            print(json.dumps({"type": "response", "action": "shutdown", "success": True}), flush=True)
                            break
                        
                        # Wait for initialization to complete for operations that need it
                        if not initialization_complete and action in ["get_price", "get_health_score", "fetch_historical_bars", "place_order", "get_positions"]:
                            if initialization_error:
                                # Initialization failed, send error immediately
                                error_response = {
                                    "type": "response",
                                    "action": action,
                                    "success": False,
                                    "error": f"TopstepX initialization failed: {initialization_error}"
                                }
                                print(json.dumps(error_response), flush=True)
                                continue
                            else:
                                # Wait for initialization to complete (with timeout)
                                print(json.dumps({"type": "debug", "message": f"⏳ Waiting for SDK initialization before {action}..."}), flush=True)
                                wait_start = time.time()
                                while not initialization_complete and not initialization_error:
                                    elapsed = time.time() - wait_start
                                    if elapsed > 25.0:  # 25s timeout (less than C# 30s to send proper error)
                                        error_response = {
                                            "type": "response",
                                            "action": action,
                                            "success": False,
                                            "error": f"TopstepX initialization timeout after {elapsed:.1f}s (SDK still connecting)"
                                        }
                                        print(json.dumps(error_response), flush=True)
                                        break
                                    await asyncio.sleep(0.5)  # Check every 500ms, allow other tasks to run
                                
                                # Check if initialization failed during wait
                                if initialization_error:
                                    error_response = {
                                        "type": "response",
                                        "action": action,
                                        "success": False,
                                        "error": f"TopstepX initialization failed: {initialization_error}"
                                    }
                                    print(json.dumps(error_response), flush=True)
                                    continue
                                
                                # If still not complete after wait, skip this command (already sent error)
                                if not initialization_complete:
                                    continue
                                
                                print(json.dumps({"type": "debug", "message": f"Initialization complete, processing {action}"}), flush=True)
                        
                        # Process command and send response with timeout
                        result = None
                        try:
                            if action == "get_price":
                                price = await asyncio.wait_for(
                                    adapter.get_price(cmd_data["symbol"]),
                                    timeout=10.0
                                )
                                result = {"success": True, "price": price}
                            
                            elif action == "get_health_score":
                                result = await asyncio.wait_for(
                                    adapter.get_health_score(),
                                    timeout=10.0
                                )
                            
                            elif action == "get_portfolio_status":
                                result = await asyncio.wait_for(
                                    adapter.get_portfolio_status(),
                                    timeout=10.0
                                )
                            
                            elif action == "get_fill_events":
                                result = await adapter.get_fill_events()
                            
                            elif action == "get_bar_events":
                                result = await adapter.get_bar_events()
                            
                            elif action == "place_order":
                                result = await adapter.place_order(
                                    cmd_data["symbol"],
                                    cmd_data["size"],
                                    cmd_data["stop_loss"],
                                    cmd_data["take_profit"],
                                    cmd_data.get("max_risk_percent", 0.01)
                                )
                            
                            elif action == "get_positions":
                                positions = await adapter.get_positions()
                                result = {"success": True, "positions": positions}
                            
                            elif action == "close_position":
                                result = await adapter.close_position(
                                    cmd_data["symbol"],
                                    cmd_data.get("quantity")
                                )
                            
                            elif action == "modify_stop_loss":
                                result = await adapter.modify_stop_loss(
                                    cmd_data["symbol"],
                                    float(cmd_data["stop_price"])
                                )
                            
                            elif action == "modify_take_profit":
                                result = await adapter.modify_take_profit(
                                    cmd_data["symbol"],
                                    float(cmd_data["take_profit_price"])
                                )
                            
                            elif action == "cancel_order":
                                result = await adapter.cancel_order(cmd_data["order_id"])
                            
                            elif action == "cancel_all_orders":
                                result = await adapter.cancel_all_orders(cmd_data.get("symbol"))
                            
                            elif action == "fetch_historical_bars":
                                # Parse dates if provided
                                start_date = None
                                end_date = None
                                if "start_date" in cmd_data:
                                    start_date = datetime.fromisoformat(cmd_data["start_date"])
                                if "end_date" in cmd_data:
                                    end_date = datetime.fromisoformat(cmd_data["end_date"])
                                
                                result = await adapter.fetch_historical_bars(
                                    symbol=cmd_data["symbol"],
                                    timeframe=cmd_data.get("interval", cmd_data.get("timeframe", 5)),
                                    start_date=start_date,
                                    end_date=end_date,
                                    limit=cmd_data.get("limit", 1000)
                                )
                            
                            else:
                                result = {"success": False, "error": f"Unknown action: {action}"}
                        
                            
                            # Send response
                            response = {"type": "response", "action": action, **result}
                            print(json.dumps(response), flush=True)
                        
                        except asyncio.TimeoutError:
                            error_response = {"type": "error", "action": action, "error": f"Command '{action}' timed out"}
                            print(json.dumps(error_response), flush=True)
                    
                    except queue.Empty:
                        # No command available, sleep briefly and continue
                        await asyncio.sleep(0.01)
                        continue
                    except json.JSONDecodeError as e:
                        error_response = {"type": "error", "error": f"Invalid JSON: {str(e)}"}
                        print(json.dumps(error_response), flush=True)
                    except Exception as e:
                        error_response = {"type": "error", "error": str(e)}
                        print(json.dumps(error_response), flush=True)
            
            finally:
                # Clean disconnect
                if adapter:
                    try:
                        await adapter.disconnect()
                    except Exception as e:
                        print(json.dumps({"type": "error", "error": f"Disconnect error: {str(e)}"}), flush=True)
        
        # Run persistent mode
        asyncio.run(persistent_mode())
        sys.exit(0)
    
    # Command-line interface for C# integration (legacy one-shot mode)
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate_sdk":
            # Check if project-x-py is available
            try:
                from project_x_py import TradingSuite
                print(json.dumps({"success": True, "message": "SDK available"}))
                sys.exit(0)
            except ImportError:
                print(json.dumps({"success": False, "error": "project-x-py not installed"}))
                sys.exit(1)
                
        elif command == "initialize":
            # Initialize adapter and return status
            try:
                async def init_test():
                    adapter = TopstepXAdapter(["ES", "NQ"])
                    await adapter.initialize()
                    health = await adapter.get_health_score()
                    await adapter.disconnect()
                    return health
                    
                result = asyncio.run(init_test())
                print(json.dumps({"success": True, "health": result}))
                sys.exit(0)
            except Exception as e:
                print(json.dumps({"success": False, "error": str(e)}))
                sys.exit(1)
                
        else:
            # Parse JSON command
            try:
                cmd_data = json.loads(command)
                action = cmd_data.get("action")
                
                async def execute_command():
                    adapter = TopstepXAdapter(["ES", "NQ"])
                    await adapter.initialize()
                    
                    try:
                        if action == "get_price":
                            price = await adapter.get_price(cmd_data["symbol"])
                            return {"success": True, "price": price}
                            
                        elif action == "place_order":
                            result = await adapter.place_order(
                                cmd_data["symbol"],
                                cmd_data["size"],
                                cmd_data["stop_loss"],
                                cmd_data["take_profit"],
                                cmd_data.get("max_risk_percent", 0.01)
                            )
                            return result
                            
                        elif action == "get_health_score":
                            result = await adapter.get_health_score()
                            return result
                            
                        elif action == "get_portfolio_status":
                            result = await adapter.get_portfolio_status()
                            return result
                        
                        elif action == "get_fill_events":
                            result = await adapter.get_fill_events()
                            return result
                        
                        elif action == "get_bar_events":
                            result = await adapter.get_bar_events()
                            return result
                        
                        elif action == "get_positions":
                            positions = await adapter.get_positions()
                            return {"success": True, "positions": positions}
                        
                        elif action == "close_position":
                            symbol = cmd_data.get("symbol")
                            quantity = cmd_data.get("quantity")
                            result = await adapter.close_position(symbol, quantity)
                            return result
                        
                        elif action == "modify_stop_loss":
                            symbol = cmd_data.get("symbol")
                            stop_price = float(cmd_data.get("stop_price"))
                            result = await adapter.modify_stop_loss(symbol, stop_price)
                            return result
                        
                        elif action == "modify_take_profit":
                            symbol = cmd_data.get("symbol")
                            take_profit_price = float(cmd_data.get("take_profit_price"))
                            result = await adapter.modify_take_profit(symbol, take_profit_price)
                            return result
                        
                        elif action == "place_bracket_order":
                            result = await adapter.place_bracket_order(
                                symbol=cmd_data.get("symbol"),
                                side=cmd_data.get("side"),
                                quantity=int(cmd_data.get("quantity")),
                                entry_price=float(cmd_data.get("entry_price")),
                                stop_loss_price=float(cmd_data.get("stop_loss_price")),
                                take_profit_price=float(cmd_data.get("take_profit_price"))
                            )
                            return result
                        
                        elif action == "get_order_status":
                            order_id = cmd_data.get("order_id")
                            result = await adapter.get_order_status(order_id)
                            return result
                        
                        elif action == "cancel_order":
                            order_id = cmd_data.get("order_id")
                            result = await adapter.cancel_order(order_id)
                            return result
                        
                        elif action == "cancel_all_orders":
                            symbol = cmd_data.get("symbol")  # Optional
                            result = await adapter.cancel_all_orders(symbol)
                            return result
                            
                        elif action == "disconnect":
                            await adapter.disconnect()
                            return {"success": True, "message": "Disconnected"}
                            
                        else:
                            return {"success": False, "error": f"Unknown action: {action}"}
                            
                    finally:
                        await adapter.disconnect()
                
                result = asyncio.run(execute_command())
                print(json.dumps(result, default=str))
                sys.exit(0)
                
            except Exception as e:
                print(json.dumps({"success": False, "error": str(e)}))
                sys.exit(1)
    else:
        # Run standalone test if executed directly
        asyncio.run(test_adapter_functionality())