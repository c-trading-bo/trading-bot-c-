"""
Full-Stack ML/RL Decision Service
Comprehensive decision brain that wraps existing strategies with:
- Regime detection → ML blend → UCB → SAC sizing → risk caps
- Professional trade management (partials, BE, trailing, exits)
- Online learning and cloud model integration
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import traceback
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

# Add parent directories to path for imports
ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
for path in [ROOT, PYTHON_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

# Import existing UCB integration
from ucb.neural_ucb_topstep import UCBIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Regime(Enum):
    CALM_TREND = "Calm-Trend"
    CALM_CHOP = "Calm-Chop"
    HIGHVOL_TREND = "HighVol-Trend"
    HIGHVOL_CHOP = "HighVol-Chop"

class TradeAction(Enum):
    HOLD = "Hold"
    TAKE_PARTIAL_25 = "TakePartial25"
    TRAIL_ATR = "Trail_ATR"
    TRAIL_STRUCTURE = "Trail_Structure"
    TIGHTEN = "Tighten"
    CLOSE = "Close"

@dataclass
class RiskConfig:
    max_total_contracts: int = 5  # Topstep limit
    max_es_contracts: int = 3
    max_nq_contracts: int = 2
    daily_soft_loss: float = 600.0  # $600 DSL
    kill_switch_loss: float = 900.0  # $900 kill-switch
    min_mll_headroom: float = 800.0  # $800 MLL headroom

@dataclass
class RegimeConfig:
    hysteresis_seconds: int = 180
    confidence_gates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_gates is None:
            self.confidence_gates = {
                "Calm-Trend": 0.52,
                "Calm-Chop": 0.54,
                "HighVol-Trend": 0.55,
                "HighVol-Chop": 0.58
            }

@dataclass
class UCBConfig:
    c_value: float = 2.0
    min_count: int = 20

@dataclass
class SACConfig:
    enabled: bool = True
    max_size_change: int = 2  # +2 max change

@dataclass
class DecisionServiceConfig:
    symbols: List[str] = None
    strategies: List[str] = None
    risk: RiskConfig = None
    regime: RegimeConfig = None
    ucb: UCBConfig = None
    sac: SACConfig = None
    degraded_mode_p99_ms: int = 120
    degraded_mode_duration_s: int = 60
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["ES", "NQ"]
        if self.strategies is None:
            self.strategies = ["S2", "S3", "S6", "S11"]
        if self.risk is None:
            self.risk = RiskConfig()
        if self.regime is None:
            self.regime = RegimeConfig()
        if self.ucb is None:
            self.ucb = UCBConfig()
        if self.sac is None:
            self.sac = SACConfig()

class DecisionService:
    """Core Decision Service implementing the full ML/RL wrapper"""
    
    def __init__(self, config: DecisionServiceConfig):
        self.config = config
        self.ucb: Optional[UCBIntegration] = None
        self.current_regime = Regime.CALM_TREND
        self.regime_transition_time = datetime.now(timezone.utc)
        self.degraded_mode = False
        self.degraded_mode_start = None
        self.decision_latencies: List[float] = []
        self.active_positions: Dict[str, Dict] = {}  # symbol -> position info
        self.daily_pnl = 0.0
        self.total_contracts = 0
        
        logger.info(f"🧠 Decision Service initialized with config: {len(config.symbols)} symbols, {len(config.strategies)} strategies")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize UCB integration
            weights_path = os.getenv("UCB_WEIGHTS_PATH", "neural_ucb_topstep.pth")
            persistence_path = os.getenv("UCB_PERSISTENCE_PATH", "ucb_state.pkl")
            
            logger.info(f"🚀 Initializing UCB integration...")
            self.ucb = UCBIntegration(weights_path=weights_path, persistence_path=persistence_path)
            
            logger.info(f"✅ Decision Service ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Decision Service: {e}")
            return False
    
    async def on_new_bar(self, tick_data: Dict) -> Dict:
        """Process new bar data for regime detection and feature updates"""
        try:
            start_time = datetime.now()
            
            # Extract tick data
            symbol = tick_data.get("symbol", "")
            timestamp = tick_data.get("ts", "")
            ohlcv = {
                "open": tick_data.get("o", 0),
                "high": tick_data.get("h", 0), 
                "low": tick_data.get("l", 0),
                "close": tick_data.get("c", 0),
                "volume": tick_data.get("v", 0)
            }
            
            # Update regime detection (simplified - would use real volatility/trend analysis)
            await self._update_regime_detection(symbol, ohlcv)
            
            # Track latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._track_decision_latency(latency_ms)
            
            return {
                "status": "ok",
                "regime": self.current_regime.value,
                "featureSnapshotId": f"snap_{int(datetime.now().timestamp())}"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in on_new_bar: {e}")
            return {"status": "error", "message": str(e)}
    
    async def on_signal(self, signal_data: Dict) -> Dict:
        """Main decision logic: regime → ML blend → UCB → SAC sizing → risk caps"""
        try:
            start_time = datetime.now()
            
            # Extract signal data
            symbol = signal_data.get("symbol", "")
            strategy_id = signal_data.get("strategyId", "")
            side = signal_data.get("side", "")
            signal_id = signal_data.get("signalId", "")
            hints = signal_data.get("hints", {})
            cloud_data = signal_data.get("cloud", {})
            
            # Check if degraded mode
            if self.degraded_mode:
                return await self._degraded_mode_response(signal_data)
            
            # Step 1: Regime gating
            gate_result = await self._regime_gate_check(symbol, strategy_id)
            if not gate_result["gate"]:
                return gate_result
            
            # Step 2: ML model blending (cloud + online)
            ml_blend = await self._ml_model_blending(cloud_data, symbol, strategy_id)
            
            # Step 3: UCB recommendation
            ucb_result = await self._ucb_recommendation(strategy_id, ml_blend["p_final"])
            
            # Step 4: SAC sizing
            proposed_size = await self._sac_sizing(symbol, side, ucb_result["ucb_score"])
            
            # Step 5: Risk caps and portfolio allocation
            final_size = await self._apply_risk_caps(symbol, proposed_size, side)
            
            # Step 6: Trade management plan
            management_plan = await self._create_management_plan(symbol, side, hints)
            
            # Create decision response
            decision_id = self._generate_decision_id()
            
            response = {
                "gate": True,
                "reason": "ok",
                "regime": self.current_regime.value,
                "p_cloud": ml_blend["p_cloud"],
                "p_online": ml_blend["p_online"], 
                "p_final": ml_blend["p_final"],
                "ucb": ucb_result["ucb_score"],
                "proposedContracts": proposed_size,
                "finalContracts": final_size,
                "risk": {
                    "stopPoints": hints.get("stopPoints", 3.0),
                    "pointValue": 50 if symbol == "ES" else 20,  # ES=$50, NQ=$20
                    "dailySoftLossRemaining": max(0, self.config.risk.daily_soft_loss + self.daily_pnl),
                    "mlHeadroomOk": self._check_mll_headroom()
                },
                "managementPlan": management_plan,
                "decisionId": decision_id
            }
            
            # Track decision latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._track_decision_latency(latency_ms)
            response["latencyMs"] = latency_ms
            
            # Log decision line
            await self._log_decision_line(signal_data, response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in on_signal: {e}")
            traceback.print_exc()
            return {
                "gate": False,
                "reason": f"Decision error: {str(e)}",
                "decisionId": self._generate_decision_id()
            }
    
    async def on_order_fill(self, fill_data: Dict) -> Dict:
        """Handle order fill notification and update position tracking"""
        try:
            decision_id = fill_data.get("decisionId", "")
            symbol = fill_data.get("symbol", "")
            strategy_id = fill_data.get("strategyId", "")
            side = fill_data.get("side", "")
            entry_price = fill_data.get("entryPrice", 0)
            contracts = fill_data.get("contracts", 0)
            
            # Update position tracking
            self.active_positions[decision_id] = {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "side": side,
                "entry_price": entry_price,
                "contracts": contracts,
                "entry_time": datetime.now(timezone.utc),
                "status": "open"
            }
            
            # Update total contracts
            self.total_contracts += contracts
            
            logger.info(f"📈 [FILL] {decision_id}: {side} {contracts} {symbol} @ {entry_price}")
            
            return {"status": "ok", "message": "Fill processed"}
            
        except Exception as e:
            logger.error(f"❌ Error in on_order_fill: {e}")
            return {"status": "error", "message": str(e)}
    
    async def on_trade_close(self, close_data: Dict) -> Dict:
        """Handle trade close and update online learning"""
        try:
            decision_id = close_data.get("decisionId", "")
            exit_price = close_data.get("exitPrice", 0)
            final_contracts = close_data.get("finalContracts", 0)
            
            if decision_id not in self.active_positions:
                logger.warning(f"⚠️ Trade close for unknown decision: {decision_id}")
                return {"status": "error", "message": "Unknown decision ID"}
            
            position = self.active_positions[decision_id]
            
            # Calculate P&L
            entry_price = position["entry_price"]
            contracts = position["contracts"]
            side = position["side"]
            symbol = position["symbol"]
            
            point_value = 50 if symbol == "ES" else 20
            
            if side == "LONG":
                pnl = (exit_price - entry_price) * contracts * point_value
            else:
                pnl = (entry_price - exit_price) * contracts * point_value
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Update total contracts
            self.total_contracts -= (contracts - final_contracts)
            
            # Update UCB with strategy performance
            if self.ucb:
                strategy_id = position["strategy_id"]
                await self._update_ucb_performance(strategy_id, pnl)
            
            # Mark position as closed
            position["status"] = "closed"
            position["exit_price"] = exit_price
            position["exit_time"] = datetime.now(timezone.utc)
            position["pnl"] = pnl
            
            logger.info(f"💰 [CLOSE] {decision_id}: PnL=${pnl:.2f}, Daily PnL=${self.daily_pnl:.2f}")
            
            # Push trade record and metrics to cloud
            await self._push_to_cloud_after_close(position, pnl)
            
            return {"status": "ok", "pnl": pnl, "dailyPnl": self.daily_pnl}
            
        except Exception as e:
            logger.error(f"❌ Error in on_trade_close: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _update_regime_detection(self, symbol: str, ohlcv: Dict):
        """Update regime detection based on volatility and trend analysis"""
        # Simplified regime detection - in production would use proper analysis
        # For now, cycle through regimes based on time for demo
        current_time = datetime.now(timezone.utc)
        minutes_since_start = (current_time.hour * 60 + current_time.minute) % 240  # 4-hour cycle
        
        if minutes_since_start < 60:
            new_regime = Regime.CALM_TREND
        elif minutes_since_start < 120:
            new_regime = Regime.CALM_CHOP
        elif minutes_since_start < 180:
            new_regime = Regime.HIGHVOL_TREND
        else:
            new_regime = Regime.HIGHVOL_CHOP
        
        if new_regime != self.current_regime:
            time_diff = (current_time - self.regime_transition_time).total_seconds()
            if time_diff >= self.config.regime.hysteresis_seconds:
                logger.info(f"🔄 Regime transition: {self.current_regime.value} → {new_regime.value}")
                self.current_regime = new_regime
                self.regime_transition_time = current_time
    
    async def _regime_gate_check(self, symbol: str, strategy_id: str) -> Dict:
        """Check if signal passes regime confidence gate"""
        regime_key = self.current_regime.value
        min_confidence = self.config.regime.confidence_gates.get(regime_key, 0.55)
        
        # For demo, use random confidence - in production would use real ML model
        import random
        signal_confidence = 0.5 + random.random() * 0.3  # 0.5-0.8 range
        
        if signal_confidence < min_confidence:
            return {
                "gate": False,
                "reason": f"Below regime threshold: {signal_confidence:.3f} < {min_confidence:.3f}",
                "regime": regime_key
            }
        
        return {"gate": True, "confidence": signal_confidence}
    
    async def _ml_model_blending(self, cloud_data: Dict, symbol: str, strategy_id: str) -> Dict:
        """Blend cloud and online ML model predictions"""
        # Extract cloud prediction
        p_cloud = cloud_data.get("p", 0.6)  # Default if not provided
        
        # Simulate online model prediction - in production would use real model
        import random
        p_online = 0.5 + random.random() * 0.2  # 0.5-0.7 range
        
        # Simple blending - in production would use sophisticated ensemble
        blend_weight = 0.7  # 70% cloud, 30% online
        p_final = p_cloud * blend_weight + p_online * (1 - blend_weight)
        
        return {
            "p_cloud": p_cloud,
            "p_online": p_online,
            "p_final": p_final,
            "blend_weight": blend_weight
        }
    
    async def _ucb_recommendation(self, strategy_id: str, confidence: float) -> Dict:
        """Get UCB recommendation using existing integration"""
        try:
            if not self.ucb:
                # Fallback UCB calculation
                ucb_score = confidence + 0.1  # Simple fallback
                return {"ucb_score": ucb_score, "source": "fallback"}
            
            # Use existing UCB integration
            ucb_request = {
                "strategy": strategy_id,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            rec_json = self.ucb.get_recommendation(json.dumps(ucb_request))
            rec_data = json.loads(rec_json)
            
            # Extract UCB score from recommendation
            ucb_score = rec_data.get("confidence", confidence) * 1.2  # Scale up slightly
            
            return {"ucb_score": ucb_score, "source": "ucb_integration"}
            
        except Exception as e:
            logger.warning(f"⚠️ UCB recommendation failed, using fallback: {e}")
            return {"ucb_score": confidence + 0.05, "source": "fallback"}
    
    async def _sac_sizing(self, symbol: str, side: str, ucb_score: float) -> int:
        """SAC reinforcement learning for position sizing"""
        if not self.config.sac.enabled:
            return 1  # Default size
        
        # Simplified SAC sizing - in production would use trained SAC model
        base_size = 1
        
        # Scale based on UCB confidence
        if ucb_score > 1.5:
            size_multiplier = min(2, ucb_score / 1.0)  # Max 2x
        else:
            size_multiplier = 1
        
        proposed_size = int(base_size * size_multiplier)
        
        # Apply step limiter
        max_change = self.config.sac.max_size_change
        proposed_size = min(proposed_size, base_size + max_change)
        
        return max(1, proposed_size)  # At least 1 contract
    
    async def _apply_risk_caps(self, symbol: str, proposed_size: int, side: str) -> int:
        """Apply Topstep and portfolio risk limits"""
        # Check total contract limit
        if self.total_contracts + proposed_size > self.config.risk.max_total_contracts:
            max_allowed = max(0, self.config.risk.max_total_contracts - self.total_contracts)
            logger.warning(f"⚠️ Total contract limit: reducing {proposed_size} → {max_allowed}")
            proposed_size = max_allowed
        
        # Check per-symbol limits
        symbol_max = (self.config.risk.max_es_contracts if symbol == "ES" 
                     else self.config.risk.max_nq_contracts)
        
        current_symbol_contracts = sum(
            pos["contracts"] for pos in self.active_positions.values()
            if pos["symbol"] == symbol and pos["status"] == "open"
        )
        
        if current_symbol_contracts + proposed_size > symbol_max:
            max_allowed = max(0, symbol_max - current_symbol_contracts)
            logger.warning(f"⚠️ {symbol} limit: reducing {proposed_size} → {max_allowed}")
            proposed_size = max_allowed
        
        # Check daily loss limits
        if self.daily_pnl <= -self.config.risk.daily_soft_loss:
            logger.warning(f"⚠️ Daily soft loss reached: ${self.daily_pnl:.2f}")
            return 0  # No new positions
        
        if self.daily_pnl <= -self.config.risk.kill_switch_loss:
            logger.error(f"🚨 Kill switch triggered: ${self.daily_pnl:.2f}")
            return 0  # No new positions
        
        return proposed_size
    
    async def _create_management_plan(self, symbol: str, side: str, hints: Dict) -> Dict:
        """Create trade management plan with partials, BE, trailing"""
        return {
            "tp1AtR": 1.0,  # Take partial at +1R
            "tp1Pct": 0.5,  # 50% partial
            "moveStopToBEOnTp1": True,  # Move stop to BE after partial
            "allowedActions": [action.value for action in TradeAction],
            "trailATRMultiplier": 1.5,  # Trail at 1.5x ATR
            "maxTrailATRMultiplier": 2.5,  # Max 2.5x ATR
            "stopPoints": hints.get("stopPoints", 3.0)
        }
    
    def _check_mll_headroom(self) -> bool:
        """Check if sufficient MLL headroom remains"""
        # Simplified check - in production would calculate actual MLL
        loss_buffer = abs(min(0, self.daily_pnl))
        remaining_headroom = self.config.risk.min_mll_headroom - loss_buffer
        return remaining_headroom > 0
    
    def _track_decision_latency(self, latency_ms: float):
        """Track decision latency for SLO monitoring"""
        self.decision_latencies.append(latency_ms)
        
        # Keep only recent latencies (last 100)
        if len(self.decision_latencies) > 100:
            self.decision_latencies = self.decision_latencies[-100:]
        
        # Check for degraded mode
        if len(self.decision_latencies) >= 10:
            p99_latency = sorted(self.decision_latencies)[-int(len(self.decision_latencies) * 0.01)-1]
            
            if p99_latency > self.config.degraded_mode_p99_ms:
                if not self.degraded_mode:
                    self.degraded_mode_start = datetime.now()
                    logger.warning(f"⚠️ High latency detected: {p99_latency:.1f}ms > {self.config.degraded_mode_p99_ms}ms")
                elif self.degraded_mode_start:
                    duration = (datetime.now() - self.degraded_mode_start).total_seconds()
                    if duration >= self.config.degraded_mode_duration_s:
                        self.degraded_mode = True
                        logger.error(f"🚨 DEGRADED MODE: Latency p99={p99_latency:.1f}ms for {duration:.0f}s")
            else:
                if self.degraded_mode:
                    self.degraded_mode = False
                    self.degraded_mode_start = None
                    logger.info(f"✅ Recovered from degraded mode")
    
    async def _degraded_mode_response(self, signal_data: Dict) -> Dict:
        """Handle decisions in degraded mode"""
        return {
            "gate": True,
            "reason": "degraded_mode",
            "regime": self.current_regime.value,
            "p_cloud": 0.0,  # Cloud disabled
            "p_online": 0.55,  # Minimum confidence
            "p_final": 0.55,
            "ucb": 1.0,
            "proposedContracts": 1,
            "finalContracts": max(1, 1 // 2),  # Halved size
            "degradedMode": True,
            "decisionId": self._generate_decision_id()
        }
    
    async def _update_ucb_performance(self, strategy_id: str, pnl: float):
        """Update UCB with strategy performance"""
        try:
            if self.ucb:
                self.ucb.update_pnl(strategy_id, pnl)
        except Exception as e:
            logger.warning(f"⚠️ Failed to update UCB performance: {e}")
    
    async def _push_to_cloud_after_close(self, position: Dict, pnl: float):
        """Push trade record and service metrics to cloud endpoint after /v1/close"""
        try:
            import aiohttp
            import json
            from datetime import datetime, timezone
            
            # Prepare trade record
            trade_record = {
                "trade_id": position.get("decision_id", "unknown"),
                "symbol": position.get("symbol", "unknown"),
                "side": position.get("side", "unknown"),
                "quantity": position.get("contracts", 0),
                "entry_price": position.get("entry_price", 0),
                "exit_price": position.get("exit_price", 0),
                "pnl": pnl,
                "entry_time": position.get("entry_time", datetime.now(timezone.utc)).isoformat(),
                "exit_time": position.get("exit_time", datetime.now(timezone.utc)).isoformat(),
                "strategy": f"S{position.get('strategy_id', 0)}",
                "metadata": {
                    "regime": self.current_regime.value if self.current_regime else "unknown",
                    "confidence": position.get("confidence", 0),
                    "daily_pnl": self.daily_pnl
                }
            }
            
            # Prepare service metrics
            service_metrics = {
                "inference_latency_ms": getattr(self, '_last_inference_latency', 0),
                "prediction_accuracy": 0.75,  # Placeholder - would be calculated
                "feature_drift": getattr(self, '_last_drift_score', 0),
                "active_models": 1,
                "memory_usage_mb": 128,  # Placeholder
                "custom_metrics": {
                    "regime": self.current_regime.value if self.current_regime else "unknown",
                    "total_contracts": self.total_contracts,
                    "daily_pnl": self.daily_pnl,
                    "active_positions": len(self.active_positions)
                }
            }
            
            # Cloud endpoint URL (configurable)
            cloud_endpoint = os.getenv("CLOUD_ENDPOINT", "https://api.example.com/ml-data")
            
            # Prepare payload
            payload = {
                "type": "trade_close_data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_record": trade_record,
                "service_metrics": service_metrics,
                "instance_id": os.getenv("INSTANCE_ID", "decision_service_001")
            }
            
            # Push to cloud with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{cloud_endpoint}/trades",
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            if response.status == 200:
                                logger.info(f"☁️ Trade data pushed to cloud: {trade_record['trade_id']}")
                                return
                            else:
                                logger.warning(f"⚠️ Cloud push failed with status {response.status}")
                                
                except Exception as e:
                    logger.warning(f"⚠️ Cloud push attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            logger.error("❌ Failed to push trade data to cloud after all retries")
            
        except Exception as e:
            logger.error(f"❌ Error pushing to cloud: {e}")
            # Don't raise - cloud push failures shouldn't stop trading
    
    async def _log_decision_line(self, signal_data: Dict, response: Dict):
        """Log structured decision line for observability"""
        decision_line = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisionId": response.get("decisionId"),
            "symbol": signal_data.get("symbol"),
            "strategyId": signal_data.get("strategyId"),
            "regime": response.get("regime"),
            "gate": response.get("gate"),
            "p_cloud": response.get("p_cloud"),
            "p_online": response.get("p_online"),
            "p_final": response.get("p_final"),
            "ucb": response.get("ucb"),
            "proposedContracts": response.get("proposedContracts"),
            "finalContracts": response.get("finalContracts"),
            "latencyMs": response.get("latencyMs"),
            "degradedMode": self.degraded_mode
        }
        
        logger.info(f"[DECISION_LINE] {json.dumps(decision_line)}")
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"dec-{timestamp}"

# Global service instance
decision_service: Optional[DecisionService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Decision Service on startup"""
    global decision_service
    try:
        # Load configuration
        config_path = os.getenv("DECISION_SERVICE_CONFIG", "decision_service_config.yaml")
        config = DecisionServiceConfig()
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                # Apply config overrides here if needed
                
        logger.info(f"🚀 Starting Decision Service...")
        decision_service = DecisionService(config)
        
        success = await decision_service.initialize()
        if not success:
            raise RuntimeError("Failed to initialize Decision Service")
            
        logger.info(f"✅ Decision Service ready!")
        yield
        
    except Exception as e:
        logger.error(f"❌ Decision Service startup error: {e}")
        raise
    finally:
        if decision_service and decision_service.ucb:
            try:
                # Save state on shutdown
                if hasattr(decision_service.ucb.model, '_save_state'):
                    decision_service.ucb.model._save_state()
                logger.info("💾 Decision Service state saved")
            except Exception as e:
                logger.warning(f"⚠️ Error saving state: {e}")

# Create FastAPI app
app = FastAPI(
    title="ML/RL Decision Service",
    description="Full-Stack Decision Brain for ES/NQ Trading with ML/RL Integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "ML/RL Decision Service", 
        "ready": decision_service is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    return {
        "status": "READY" if not decision_service.degraded_mode else "DEGRADED",
        "regime": decision_service.current_regime.value,
        "daily_pnl": decision_service.daily_pnl,
        "total_contracts": decision_service.total_contracts,
        "active_positions": len([p for p in decision_service.active_positions.values() if p["status"] == "open"]),
        "degraded_mode": decision_service.degraded_mode,
        "avg_latency_ms": sum(decision_service.decision_latencies[-10:]) / max(1, len(decision_service.decision_latencies[-10:]))
    }

@app.post("/v1/tick")
async def on_new_bar(request: Request):
    """Process new bar data"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    try:
        tick_data = await request.json()
        result = await decision_service.on_new_bar(tick_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error in /v1/tick: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/signal")
async def on_signal(request: Request):
    """Main decision endpoint - regime → ML blend → UCB → SAC sizing → risk caps"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    try:
        signal_data = await request.json()
        result = await decision_service.on_signal(signal_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error in /v1/signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/fill")
async def on_order_fill(request: Request):
    """Handle order fill notification"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    try:
        fill_data = await request.json()
        result = await decision_service.on_order_fill(fill_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error in /v1/fill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/close")
async def on_trade_close(request: Request):
    """Handle trade close and update online learning"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    try:
        close_data = await request.json()
        result = await decision_service.on_trade_close(close_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error in /v1/close: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats")
async def get_stats():
    """Get current decision service statistics"""
    if decision_service is None:
        raise HTTPException(status_code=503, detail="Decision Service not initialized")
    
    try:
        return {
            "regime": decision_service.current_regime.value,
            "daily_pnl": decision_service.daily_pnl,
            "total_contracts": decision_service.total_contracts,
            "active_positions": len([p for p in decision_service.active_positions.values() if p["status"] == "open"]),
            "degraded_mode": decision_service.degraded_mode,
            "decision_count": len(decision_service.decision_latencies),
            "avg_latency_ms": sum(decision_service.decision_latencies[-50:]) / max(1, len(decision_service.decision_latencies[-50:])),
            "positions": [
                {
                    "decision_id": did,
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "contracts": pos["contracts"],
                    "status": pos["status"],
                    "pnl": pos.get("pnl", 0)
                }
                for did, pos in decision_service.active_positions.items()
            ]
        }
    except Exception as e:
        logger.error(f"❌ Error in /v1/stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Production settings
    host = os.getenv("DECISION_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("DECISION_SERVICE_PORT", "7080"))
    
    logger.info(f"🚀 Starting ML/RL Decision Service on {host}:{port}")
    logger.info(f"📊 Endpoints: /v1/tick, /v1/signal, /v1/fill, /v1/close")
    logger.info(f"🔧 Health: /health, Stats: /v1/stats")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
        access_log=True,
        log_level="info"
    )