#!/usr/bin/env python3
"""
Simplified Decision Service for testing without external dependencies
Provides the core ML/RL decision logic using only Python standard library
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Regime(Enum):
    CALM_TREND = "Calm-Trend"
    CALM_CHOP = "Calm-Chop"
    HIGHVOL_TREND = "HighVol-Trend"
    HIGHVOL_CHOP = "HighVol-Chop"

@dataclass
class RiskConfig:
    max_total_contracts: int = 5
    max_es_contracts: int = 3
    max_nq_contracts: int = 2
    daily_soft_loss: float = 600.0
    kill_switch_loss: float = 900.0

@dataclass
class RegimeConfig:
    confidence_gates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_gates is None:
            self.confidence_gates = {
                "Calm-Trend": 0.52,
                "Calm-Chop": 0.54,
                "HighVol-Trend": 0.55,
                "HighVol-Chop": 0.58
            }

class SimpleDecisionService:
    """Simplified Decision Service for testing"""
    
    def __init__(self):
        self.risk_config = RiskConfig()
        self.regime_config = RegimeConfig()
        self.current_regime = Regime.CALM_TREND
        self.daily_pnl = 0.0
        self.total_contracts = 0
        self.active_positions = {}
        self.degraded_mode = False
        
        logger.info("🧠 Simple Decision Service initialized")
    
    def on_new_bar(self, tick_data: Dict) -> Dict:
        """Process new bar data"""
        # Update regime based on time
        current_time = datetime.now(timezone.utc)
        minutes = (current_time.hour * 60 + current_time.minute) % 240
        
        if minutes < 60:
            self.current_regime = Regime.CALM_TREND
        elif minutes < 120:
            self.current_regime = Regime.CALM_CHOP
        elif minutes < 180:
            self.current_regime = Regime.HIGHVOL_TREND
        else:
            self.current_regime = Regime.HIGHVOL_CHOP
        
        return {
            "status": "ok",
            "regime": self.current_regime.value,
            "featureSnapshotId": f"snap_{int(time.time())}"
        }
    
    def on_signal(self, signal_data: Dict) -> Dict:
        """Main decision logic"""
        import random
        
        symbol = signal_data.get("symbol", "")
        strategy_id = signal_data.get("strategyId", "")
        side = signal_data.get("side", "")
        cloud_data = signal_data.get("cloud", {})
        
        # Regime gating
        regime_key = self.current_regime.value
        min_confidence = self.regime_config.confidence_gates.get(regime_key, 0.55)
        signal_confidence = 0.5 + random.random() * 0.3
        
        if signal_confidence < min_confidence:
            return {
                "gate": False,
                "reason": f"Below regime threshold: {signal_confidence:.3f} < {min_confidence:.3f}",
                "regime": regime_key,
                "decisionId": f"dec-{int(time.time())}"
            }
        
        # ML blending
        p_cloud = cloud_data.get("p", 0.6)
        p_online = 0.5 + random.random() * 0.2
        p_final = p_cloud * 0.7 + p_online * 0.3
        
        # UCB score
        ucb_score = p_final * 1.2
        
        # Sizing
        proposed_size = 1 if ucb_score > 0.6 else 2 if ucb_score > 0.8 else 1
        
        # Risk caps
        final_size = min(proposed_size, self.risk_config.max_total_contracts - self.total_contracts)
        if symbol == "ES":
            final_size = min(final_size, self.risk_config.max_es_contracts)
        else:
            final_size = min(final_size, self.risk_config.max_nq_contracts)
        
        final_size = max(0, final_size)
        
        decision_id = f"dec-{int(time.time() * 1000)}"
        
        return {
            "gate": final_size > 0,
            "reason": "ok" if final_size > 0 else "size_limited",
            "regime": self.current_regime.value,
            "p_cloud": p_cloud,
            "p_online": p_online,
            "p_final": p_final,
            "ucb": ucb_score,
            "proposedContracts": proposed_size,
            "finalContracts": final_size,
            "risk": {
                "stopPoints": 3.0,
                "pointValue": 50 if symbol == "ES" else 20,
                "dailySoftLossRemaining": max(0, self.risk_config.daily_soft_loss + self.daily_pnl),
                "mlHeadroomOk": True
            },
            "managementPlan": {
                "tp1AtR": 1.0,
                "tp1Pct": 0.5,
                "moveStopToBEOnTp1": True,
                "allowedActions": ["Hold", "TakePartial25", "Trail_ATR", "Close"],
                "trailATRMultiplier": 1.5,
                "stopPoints": 3.0
            },
            "decisionId": decision_id,
            "latencyMs": 15.5,
            "degradedMode": self.degraded_mode
        }
    
    def on_order_fill(self, fill_data: Dict) -> Dict:
        """Handle order fill"""
        decision_id = fill_data.get("decisionId", "")
        contracts = fill_data.get("contracts", 0)
        
        self.total_contracts += contracts
        self.active_positions[decision_id] = fill_data
        
        logger.info(f"📈 Fill processed: {decision_id} - {contracts} contracts")
        
        return {"status": "ok", "message": "Fill processed"}
    
    def on_trade_close(self, close_data: Dict) -> Dict:
        """Handle trade close"""
        decision_id = close_data.get("decisionId", "")
        exit_price = close_data.get("exitPrice", 0)
        
        if decision_id in self.active_positions:
            fill_data = self.active_positions[decision_id]
            entry_price = fill_data.get("entryPrice", 0)
            contracts = fill_data.get("contracts", 0)
            side = fill_data.get("side", "LONG")
            
            # Calculate P&L
            if side == "LONG":
                pnl = (exit_price - entry_price) * contracts * 50
            else:
                pnl = (entry_price - exit_price) * contracts * 50
            
            self.daily_pnl += pnl
            self.total_contracts -= contracts
            
            del self.active_positions[decision_id]
            
            logger.info(f"💰 Close processed: {decision_id} - P&L: ${pnl:.2f}")
            
            return {"status": "ok", "pnl": pnl, "dailyPnl": self.daily_pnl}
        else:
            return {"status": "error", "message": "Unknown decision ID"}
    
    def get_health(self) -> Dict:
        """Get health status"""
        return {
            "status": "READY",
            "regime": self.current_regime.value,
            "daily_pnl": self.daily_pnl,
            "total_contracts": self.total_contracts,
            "active_positions": len(self.active_positions),
            "degraded_mode": self.degraded_mode,
            "avg_latency_ms": 15.5
        }
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "regime": self.current_regime.value,
            "daily_pnl": self.daily_pnl,
            "total_contracts": self.total_contracts,
            "active_positions": len(self.active_positions),
            "degraded_mode": self.degraded_mode,
            "decision_count": 10,
            "avg_latency_ms": 15.5,
            "positions": [
                {
                    "decision_id": decision_id,
                    "symbol": pos.get("symbol", ""),
                    "side": pos.get("side", ""),
                    "contracts": pos.get("contracts", 0),
                    "status": "open",
                    "pnl": 0
                }
                for decision_id, pos in self.active_positions.items()
            ]
        }

# Global service instance
decision_service = SimpleDecisionService()

class DecisionServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Decision Service"""
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == "/":
            self._send_json_response(200, {
                "service": "ML/RL Decision Service (Simple)",
                "ready": True,
                "version": "1.0.0-simple"
            })
        elif path == "/health":
            self._send_json_response(200, decision_service.get_health())
        elif path == "/v1/stats":
            self._send_json_response(200, decision_service.get_stats())
        else:
            self._send_json_response(404, {"error": "Not found"})
    
    def do_POST(self):
        """Handle POST requests"""
        path = urlparse(self.path).path
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data) if post_data else {}
            
            if path == "/v1/tick":
                response = decision_service.on_new_bar(data)
                self._send_json_response(200, response)
            elif path == "/v1/signal":
                response = decision_service.on_signal(data)
                self._send_json_response(200, response)
            elif path == "/v1/fill":
                response = decision_service.on_order_fill(data)
                self._send_json_response(200, response)
            elif path == "/v1/close":
                response = decision_service.on_trade_close(data)
                self._send_json_response(200, response)
            else:
                self._send_json_response(404, {"error": "Not found"})
                
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            self._send_json_response(500, {"error": str(e)})
    
    def _send_json_response(self, status_code: int, data: Dict):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2).encode('utf-8')
        self.wfile.write(json_data)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log message"""
        logger.info(f"[HTTP] {format % args}")

def run_server(host="127.0.0.1", port=7080):
    """Run the Decision Service HTTP server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DecisionServiceHandler)
    
    logger.info(f"🚀 Simple Decision Service starting on {host}:{port}")
    logger.info(f"📊 Endpoints: /v1/tick, /v1/signal, /v1/fill, /v1/close")
    logger.info(f"🔧 Health: /health, Stats: /v1/stats")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopping...")
        httpd.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Decision Service')
    parser.add_argument('--input', type=str, help='Input data for decision')
    parser.add_argument('--host', type=str, default=os.getenv("DECISION_SERVICE_HOST", "127.0.0.1"), 
                       help='Server host')
    parser.add_argument('--port', type=int, default=int(os.getenv("DECISION_SERVICE_PORT", "7080")), 
                       help='Server port')
    parser.add_argument('--server', action='store_true', help='Run as HTTP server')
    
    args = parser.parse_args()
    
    if args.input:
        # Command line mode - used by C# Python integration
        try:
            input_data = json.loads(args.input) if args.input.startswith('{') else {"input": args.input}
            result = decision_service.on_signal(input_data)
            
            # Return the decision in a simple format
            if result.get("gate", False):
                decision = f"BUY" if result.get("finalContracts", 0) > 0 else "HOLD"
            else:
                decision = "HOLD"
                
            print(decision)
        except Exception as e:
            print("HOLD")  # Safe fallback
            sys.exit(1)
    else:
        # Server mode
        run_server(args.host, args.port)