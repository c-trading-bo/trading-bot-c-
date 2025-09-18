import os, json, numpy as np, hashlib, pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import torch, torch.nn as nn
from datetime import datetime
import sys
import asyncio

# Add SDK bridge for market data access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sdk_bridge import SDKBridge

@dataclass
class TopStepConfig:
    ACCOUNT_SIZE = 50_000
    MAX_DRAWDOWN = 2_000
    DAILY_LOSS_LIMIT = 1_000
    TRAILING_STOP = 48_000
    ES_POINT_VALUE = 50
    NQ_POINT_VALUE = 20
    RISK_PER_TRADE = 0.01  # 1% = $500 baseline
    EXPLORATION_BONUS = 0.3
    CONFIDENCE_THRESHOLD = 0.65

class NeuralUCBTopStep(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=128, persistence_path=None):
        super().__init__()
        self.config = TopStepConfig()
        self.conf_temp = 2.0  # Temperature for confidence scaling
        self.persistence_path = persistence_path or "ucb_state.pkl"

        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.uncertainty_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus()
        )

        # State that persists across restarts
        self.strategy_stats: Dict[int, Dict] = {}
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.account_balance = float(self.config.ACCOUNT_SIZE)
        self.last_save = datetime.now()
        
        # Load persisted state if available
        self._load_state()

    def _stable_sid(self, name: str) -> int:
        """Generate consistent strategy ID across Python restarts"""
        return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), 'little')

    def _load_state(self):
        """Load persisted state from disk"""
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'rb') as f:
                    state = pickle.load(f)
                    self.strategy_stats = state.get('strategy_stats', {})
                    self.current_drawdown = state.get('current_drawdown', 0.0)
                    self.daily_pnl = state.get('daily_pnl', 0.0)
                    self.account_balance = state.get('account_balance', float(self.config.ACCOUNT_SIZE))
                    print(f"📁 [UCB] Loaded state: {len(self.strategy_stats)} strategies, PnL: ${self.daily_pnl:.2f}")
        except Exception as e:
            print(f"⚠️ [UCB] Failed to load state: {e}")

    def _save_state(self):
        """Save state to disk (called periodically)"""
        try:
            state = {
                'strategy_stats': self.strategy_stats,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'account_balance': self.account_balance,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(state, f)
            self.last_save = datetime.now()
        except Exception as e:
            print(f"⚠️ [UCB] Failed to save state: {e}")

    def _clamp_inputs(self, market_features: Dict) -> Dict:
        """Server-side input validation and clamping"""
        if not market_features:
            return {}
        
        clamped = {}
        for key, value in market_features.items():
            try:
                val = float(value) if value is not None else 0.0
                
                # Apply sensible bounds
                if key in ['es_atr', 'nq_atr']:
                    clamped[key] = max(0.25, min(val, 100.0))  # ATR bounds
                elif key in ['vix']:
                    clamped[key] = max(5.0, min(val, 100.0))   # VIX bounds
                elif key in ['tick']:
                    clamped[key] = max(-3000, min(val, 3000))  # TICK bounds
                elif key in ['add']:
                    clamped[key] = max(-2000, min(val, 2000))  # ADD bounds
                elif key in ['correlation']:
                    # Guard against NaN correlation
                    val = np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0)
                    clamped[key] = max(-1.0, min(val, 1.0))    # Correlation bounds
                elif key in ['rsi_es', 'rsi_nq']:
                    clamped[key] = max(0.0, min(val, 100.0))   # RSI bounds
                else:
                    # For prices, volumes, etc - basic NaN protection
                    clamped[key] = np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)
                    
            except (ValueError, TypeError):
                clamped[key] = 0.0  # Default fallback
                
        return clamped

    def calculate_position_size(self, confidence: float, instrument: str, 
                               current_drawdown: float, market_features: Dict = None) -> int:
        base_risk = self.account_balance * self.config.RISK_PER_TRADE

        drawdown_ratio = current_drawdown / self.config.MAX_DRAWDOWN if self.config.MAX_DRAWDOWN else 0.0
        if drawdown_ratio > 0.75:
            risk_multiplier = 0.25
        elif drawdown_ratio > 0.5:
            risk_multiplier = 0.5
        else:
            risk_multiplier = 1.0

        confidence_multiplier = min(max(confidence - 0.5, 0.0) * 2.0, 1.0)
        risk_amount = base_risk * risk_multiplier * confidence_multiplier

        # Dynamic stops with safety bounds (clamped inputs)
        if market_features:
            clamped = self._clamp_inputs(market_features)
            es_stop = float(clamped.get('es_atr', 10.0))
            nq_stop = float(clamped.get('nq_atr', 25.0))
        else:
            es_stop, nq_stop = 10.0, 25.0

        # Support MES/MNQ micro contracts
        if instrument.upper() in ["NQ", "MNQ"]:
            per_contract_risk = nq_stop * (self.config.NQ_POINT_VALUE if instrument.upper() == "NQ" else 2)
        else:  # ES, MES
            per_contract_risk = es_stop * (self.config.ES_POINT_VALUE if instrument.upper() == "ES" else 5)

        contracts = int(risk_amount // per_contract_risk) if per_contract_risk > 0 else 0
        max_contracts = 3 if current_drawdown < 500 else 2 if current_drawdown < 1000 else 1
        return max(0, min(contracts, max_contracts))

    @torch.inference_mode()
    def forward(self, features: torch.Tensor, strategy_id: int, 
                instrument: str, market_features: Dict = None) -> Tuple[float, float, int]:
        value = self.value_network(features).squeeze()
        uncertainty = self.uncertainty_network(features).squeeze()
        ucb_score = value + self.config.EXPLORATION_BONUS * torch.sqrt(uncertainty + 1e-8)
        confidence = torch.sigmoid(ucb_score / self.conf_temp).item()

        if confidence < self.config.CONFIDENCE_THRESHOLD:
            return 0.0, confidence, 0

        position_size = self.calculate_position_size(
            confidence, instrument, self.current_drawdown, market_features
        )
        return float(ucb_score.item()), float(confidence), int(position_size)

    def update_strategy_performance(self, strategy_id: int, pnl: float):
        s = self.strategy_stats.setdefault(strategy_id, {
            'trades': 0, 'total_pnl': 0.0, 'wins': 0, 'losses': 0,
            'avg_win': 0.0, 'avg_loss': 0.0, 'last_10_trades': []
        })
        s['trades'] += 1
        s['total_pnl'] += pnl
        if pnl > 0:
            s['wins'] += 1
            s['avg_win'] = (s['avg_win'] * (s['wins'] - 1) + pnl) / s['wins']
        else:
            s['losses'] += 1
            s['avg_loss'] = (s['avg_loss'] * (s['losses'] - 1) + pnl) / s['losses']

        s['last_10_trades'].append(pnl)
        if len(s['last_10_trades']) > 10:
            s['last_10_trades'].pop(0)

        self.daily_pnl += pnl
        self.account_balance += pnl
        if self.daily_pnl < 0:
            self.current_drawdown = max(self.current_drawdown, abs(self.daily_pnl))

        # Auto-save state every 10 trades or every 5 minutes
        if (s['trades'] % 10 == 0) or (datetime.now() - self.last_save).seconds > 300:
            self._save_state()

    def should_stop_trading(self) -> Tuple[bool, str, str]:
        if self.daily_pnl <= -self.config.DAILY_LOSS_LIMIT:
            return True, f"Daily loss limit reached: ${self.daily_pnl:.2f}", "hard_stop"
        if self.current_drawdown >= self.config.MAX_DRAWDOWN:
            return True, f"Max drawdown reached: ${self.current_drawdown:.2f}", "hard_stop"
        if self.account_balance <= self.config.TRAILING_STOP:
            return True, f"Account below minimum: ${self.account_balance:.2f}", "hard_stop"
        if self.daily_pnl <= -(self.config.DAILY_LOSS_LIMIT * 0.9):
            return False, f"Near daily loss limit: ${self.daily_pnl:.2f}", "warning"
        return False, "OK", ""

    def get_strategy_recommendation(self, market_features: Dict, 
                                   available_strategies: List[str]) -> Dict:
        # Clamp inputs first
        market_features = self._clamp_inputs(market_features)
        
        trade_blocked, msg, level = self.should_stop_trading()
        result = {
            'trade': False,
            'reason': None,
            'warning': msg if level == "warning" else None,
            'strategy': None,
            'position_size': 0
        }
        if trade_blocked:
            result['reason'] = msg
            return result

        instrument = (market_features.get('instrument') or 'ES').upper()
        # Normalize micro contracts
        if instrument in ['MES', 'MNQ']:
            instrument = 'ES' if instrument == 'MES' else 'NQ'
        
        best = {'score': -float('inf'), 'strategy': None, 'pos': 0, 'conf': 0.0}

        with torch.inference_mode():
            for strategy in available_strategies:
                sid = self._stable_sid(strategy)
                feats = self._prepare_features(market_features, strategy)
                u, c, p = self.forward(feats, sid, instrument, market_features)
                if u > best['score'] and p > 0:
                    best.update(score=u, strategy=strategy, pos=p, conf=c)

        if best['strategy']:
            return {
                'trade': True,
                'strategy': best['strategy'],
                'confidence': best['conf'],
                'position_size': best['pos'],
                'ucb_score': best['score'],
                'risk_amount': best['pos'] * 500,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'warning': result['warning']
            }
        result['reason'] = 'No strategy meets confidence threshold'
        return result

    def _prepare_features(self, market_data: Dict, strategy: str) -> torch.Tensor:
        features: List[float] = [
            float(market_data.get('es_price', 0) or 0),
            float(market_data.get('nq_price', 0) or 0),
            float(market_data.get('es_volume', 0) or 0),
            float(market_data.get('nq_volume', 0) or 0),
            float(market_data.get('vix', 0) or 0),
            float(market_data.get('tick', 0) or 0),
            float(market_data.get('add', 0) or 0),
            float(market_data.get('correlation', 0) or 0),  # Already clamped
            float(market_data.get('rsi_es', 0) or 0),
            float(market_data.get('rsi_nq', 0) or 0),
        ]
        
        sid = self._stable_sid(strategy)
        features.append((sid % 1000) / 1000.0)

        if sid in self.strategy_stats:
            stats = self.strategy_stats[sid]
            win_rate = stats['wins'] / max(stats['trades'], 1)
            avg_pnl = stats['total_pnl'] / max(stats['trades'], 1)
            recent = float(np.mean(stats['last_10_trades'])) if stats['last_10_trades'] else 0.0
            features.extend([win_rate, avg_pnl, recent])
        else:
            features.extend([0.5, 0.0, 0.0])

        features.extend([
            self.current_drawdown / self.config.MAX_DRAWDOWN if self.config.MAX_DRAWDOWN else 0.0,
            self.daily_pnl / self.config.DAILY_LOSS_LIMIT if self.config.DAILY_LOSS_LIMIT else 0.0,
            self.account_balance / self.config.ACCOUNT_SIZE if self.config.ACCOUNT_SIZE else 0.0
        ])

        if len(features) < 50:
            features.extend([0.0] * (50 - len(features)))
        
        # Final NaN guard on all features
        arr = np.asarray(features[:50], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.from_numpy(arr).unsqueeze(0)

    def reset_daily_stats(self):
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self._save_state()  # Persist the reset

class UCBIntegration:
    """
    UCB integration with SDK bridge for live market data and order execution.
    
    This replaces legacy market data sources with the SDK adapter.
    """
    
    def __init__(self, instruments: List[str] = None):
        """Initialize UCB integration with SDK bridge."""
        self.instruments = instruments or ['MNQ', 'ES']
        self.sdk_bridge = SDKBridge(self.instruments)
        self.model = NeuralUCBTopStep()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize SDK bridge connection."""
        try:
            success = await self.sdk_bridge.initialize()
            self._initialized = success
            return success
        except Exception as e:
            print(f"UCB Integration initialization failed: {e}")
            return False
    
    async def get_live_market_features(self) -> Dict[str, Any]:
        """
        Get live market features from SDK bridge.
        
        This replaces legacy market data fetching methods.
        """
        if not self._initialized:
            raise RuntimeError("UCB Integration not initialized")
        
        try:
            # Get live prices for all instruments
            market_data = {}
            
            for instrument in self.instruments:
                try:
                    price = await self.sdk_bridge.get_live_price(instrument)
                    if instrument.upper() in ['ES', 'MES']:
                        market_data['es_price'] = price
                    elif instrument.upper() in ['NQ', 'MNQ']:
                        market_data['nq_price'] = price
                except Exception as e:
                    print(f"Failed to get price for {instrument}: {e}")
                    # Set default values for missing data
                    if instrument.upper() in ['ES', 'MES']:
                        market_data['es_price'] = 4500.0
                    elif instrument.upper() in ['NQ', 'MNQ']:
                        market_data['nq_price'] = 18500.0
            
            # Get historical data for indicators (using SDK bridge)
            try:
                # Get recent bars for technical indicators
                es_bars = await self.sdk_bridge.get_historical_bars('ES', '1m', 20)
                nq_bars = await self.sdk_bridge.get_historical_bars('MNQ', '1m', 20)
                
                # Calculate basic indicators from historical data
                if es_bars:
                    market_data['es_volume'] = es_bars[-1].get('volume', 100)
                    # Simple RSI calculation (simplified)
                    closes = [bar.get('close', market_data.get('es_price', 4500)) for bar in es_bars[-14:]]
                    market_data['rsi_es'] = self._calculate_simple_rsi(closes)
                else:
                    market_data['es_volume'] = 100
                    market_data['rsi_es'] = 50.0
                
                if nq_bars:
                    market_data['nq_volume'] = nq_bars[-1].get('volume', 100)
                    closes = [bar.get('close', market_data.get('nq_price', 18500)) for bar in nq_bars[-14:]]
                    market_data['rsi_nq'] = self._calculate_simple_rsi(closes)
                else:
                    market_data['nq_volume'] = 100
                    market_data['rsi_nq'] = 50.0
                    
            except Exception as e:
                print(f"Failed to get historical data for indicators: {e}")
                # Set default values
                market_data.update({
                    'es_volume': 100,
                    'nq_volume': 100,
                    'rsi_es': 50.0,
                    'rsi_nq': 50.0
                })
            
            # Add additional market features with defaults
            market_data.update({
                'vix': 20.0,  # Would need VIX data feed
                'tick': 0.0,  # Would need tick data
                'add': 0.0,   # Would need ADD data
                'correlation': 0.8,  # ES/NQ correlation
                'timestamp': datetime.now().isoformat()
            })
            
            return market_data
            
        except Exception as e:
            print(f"Failed to get live market features: {e}")
            # Return minimal default data
            return {
                'es_price': 4500.0,
                'nq_price': 18500.0,
                'es_volume': 100,
                'nq_volume': 100,
                'vix': 20.0,
                'tick': 0.0,
                'add': 0.0,
                'correlation': 0.8,
                'rsi_es': 50.0,
                'rsi_nq': 50.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_simple_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate simple RSI from price closes."""
        if len(closes) < 2:
            return 50.0
            
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) == 0:
            return 50.0
            
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return max(0.0, min(100.0, rsi))
    
    async def get_strategy_recommendation_live(self, available_strategies: List[str]) -> Dict[str, Any]:
        """
        Get strategy recommendation using live market data from SDK bridge.
        
        This is the main entry point for live trading decisions.
        """
        if not self._initialized:
            raise RuntimeError("UCB Integration not initialized")
        
        try:
            # Get live market features
            market_features = await self.get_live_market_features()
            
            # Get strategy recommendation
            recommendation = self.model.get_strategy_recommendation(market_features, available_strategies)
            
            # Add market context
            recommendation['market_features'] = market_features
            recommendation['timestamp'] = datetime.now().isoformat()
            
            return recommendation
            
        except Exception as e:
            print(f"Failed to get strategy recommendation: {e}")
            return {
                'trade': False,
                'reason': f'Error getting recommendation: {str(e)}',
                'strategy': None,
                'position_size': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def place_order_via_sdk(
        self,
        symbol: str,
        size: int,
        stop_loss: float,
        take_profit: float,
        max_risk_percent: float = 0.01
    ) -> Dict[str, Any]:
        """
        Place order via SDK bridge with risk management.
        
        This replaces legacy order placement methods.
        """
        if not self._initialized:
            raise RuntimeError("UCB Integration not initialized")
        
        return await self.sdk_bridge.place_order(
            symbol=symbol,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_risk_percent=max_risk_percent
        )
    
    async def get_account_state(self) -> Dict[str, Any]:
        """Get account state via SDK bridge."""
        if not self._initialized:
            raise RuntimeError("UCB Integration not initialized")
        
        return await self.sdk_bridge.get_account_state()
    
    async def disconnect(self):
        """Clean disconnect from SDK bridge."""
        await self.sdk_bridge.disconnect()
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Keep the original class for backward compatibility
class UCBIntegrationLegacy:
    def __init__(self, weights_path: str = None, persistence_path: str = None):
        torch.set_num_threads(1)  # Single process for consistency
        self.model = NeuralUCBTopStep(persistence_path=persistence_path)
        self.model.eval()

        path = weights_path
        if not path:
            here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
            path = os.path.join(here, "neural_ucb_topstep.pth")
        if os.path.exists(path):
            try:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.0
                except TypeError:
                    state = torch.load(path, map_location="cpu")                     # torch<2.0
                self.model.load_state_dict(state, strict=False)
                print(f"✅ [UCB] Loaded model weights from {path}")
            except Exception as e:
                print(f"⚠️ [UCB] Failed to load weights: {e}")

    def get_recommendation(self, market_data_json: str) -> str:
        try:
            market_data = json.loads(market_data_json or "{}")
            
            # Map your existing strategies to UCB strategies
            strategies = [
                "S2_mean_reversion",      # Your S2 strategy
                "S3_compression_breakout", # Your S3 strategy  
                "S6_opening_drive",       # Your S6 strategy
                "S11_frequent_trade"      # Your S11 strategy
            ]
            
            rec = self.model.get_strategy_recommendation(market_data, strategies)
            return json.dumps(rec, separators=(",", ":"))
        except Exception as e:
            print(f"❌ [UCB] Error in get_recommendation: {e}")
            return json.dumps({"trade": False, "reason": f"Error: {str(e)}"})
    
    def update_pnl(self, strategy: str, pnl: float) -> str:
        """Called from C# after each trade completes"""
        try:
            sid = self.model._stable_sid(strategy)
            self.model.update_strategy_performance(sid, float(pnl))
            return json.dumps({"status": "ok", "updated_strategy": strategy, "pnl": pnl})
        except Exception as e:
            print(f"❌ [UCB] Error updating P&L: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    
    def reset_daily(self) -> str:
        """Called at start of each trading day"""
        try:
            self.model.reset_daily_stats()
            return json.dumps({"status": "reset", "timestamp": datetime.now().isoformat()})
        except Exception as e:
            print(f"❌ [UCB] Error resetting daily: {e}")
            return json.dumps({"status": "error", "message": str(e)})
