#!/usr/bin/env python3
"""
Vendor Data Feature Generator for 24/7 Learning

Generates synthetic training data from vendor historical bars when local bot is offline.
Maintains same feature columns as real candidates.parquet to keep training active.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional TA-Lib imports - graceful fallback if not available
try:
    import ta
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import EMAIndicator, SMAIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA-Lib not available, using simplified indicators")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class VendorFeatureGenerator:
    """Generates training features from vendor historical data."""
    
    def __init__(self, vendor_dir: str = "data/vendor", output_path: str = "data/logs/candidates.vendor.parquet"):
        self.vendor_dir = Path(vendor_dir)
        self.output_path = Path(output_path)
        self.min_bars = 200  # Minimum bars needed for indicators
        
    def load_vendor_data(self, symbol: str = "ES") -> Optional[pd.DataFrame]:
        """Load vendor historical bars."""
        try:
            # Look for current month file
            current_month = datetime.now().strftime("%Y-%m")
            file_pattern = f"{symbol}/{current_month}*.parquet"
            
            vendor_files = list(self.vendor_dir.glob(file_pattern))
            if not vendor_files:
                # Fallback to any recent file
                vendor_files = list(self.vendor_dir.glob(f"{symbol}/*.parquet"))
                vendor_files.sort(reverse=True)  # Most recent first
                
            if not vendor_files:
                logger.warning(f"No vendor data found for {symbol} in {self.vendor_dir}")
                return None
                
            logger.info(f"Loading vendor data from: {vendor_files[0]}")
            df = pd.read_parquet(vendor_files[0])
            
            # Ensure required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None
                
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Loaded {len(df)} bars from {df.timestamp.min()} to {df.timestamp.max()}")
            return df
            
        except Exception as ex:
            logger.error(f"Failed to load vendor data: {ex}")
            return None
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to vendor data."""
        try:
            # Price-based indicators
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
            
            # Volatility indicators
            bb = BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            
            # Volume indicators
            df['volume_sma'] = SMAIndicator(df['volume'], window=20).sma_indicator()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_change'] = df['close'].pct_change()
            df['price_momentum'] = df['close'].rolling(10).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
            
            # Market microstructure approximations (since we don't have Level 2 data)
            df['spread_estimate'] = (df['high'] - df['low']) / df['close'] * 10000  # Basis points
            df['tick_direction'] = np.sign(df['close'].diff())
            
            logger.info(f"Computed technical indicators for {len(df)} bars")
            return df
            
        except Exception as ex:
            logger.error(f"Failed to compute indicators: {ex}")
            return df
    
    def identify_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market regimes (Range, Trend, Vol)."""
        try:
            # Trend strength (EMA slope)
            df['ema20_slope'] = df['ema20'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Volatility regime
            df['vol_rolling'] = df['price_change'].rolling(20).std()
            df['vol_zscore'] = (df['vol_rolling'] - df['vol_rolling'].rolling(100).mean()) / df['vol_rolling'].rolling(100).std()
            
            # Range vs Trend classification
            conditions = [
                (abs(df['ema20_slope']) < 0.1) & (df['vol_zscore'] < 0.5),  # Range
                (abs(df['ema20_slope']) >= 0.1),  # Trend
                (df['vol_zscore'] >= 0.5)  # High Vol
            ]
            choices = ['Range', 'Trend', 'Vol']
            df['regime'] = np.select(conditions, choices, default='Range')
            
            logger.info(f"Regime distribution: {df['regime'].value_counts().to_dict()}")
            return df
            
        except Exception as ex:
            logger.error(f"Failed to identify regimes: {ex}")
            df['regime'] = 'Range'  # Default fallback
            return df
    
    def generate_synthetic_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate synthetic trading signals for 4 strategies."""
        signals = []
        
        for i in range(self.min_bars, len(df)):
            bar = df.iloc[i]
            
            # EMA Cross Strategy
            if bar['ema20'] > bar['ema50'] and df.iloc[i-1]['ema20'] <= df.iloc[i-1]['ema50']:
                signals.append({
                    'idx': i,
                    'strategy': 'EmaCrossStrategy',
                    'side': 'BUY',
                    'signal_strength': min(abs(bar['ema20'] - bar['ema50']) / bar['close'], 1.0)
                })
            elif bar['ema20'] < bar['ema50'] and df.iloc[i-1]['ema20'] >= df.iloc[i-1]['ema50']:
                signals.append({
                    'idx': i,
                    'strategy': 'EmaCrossStrategy', 
                    'side': 'SELL',
                    'signal_strength': min(abs(bar['ema20'] - bar['ema50']) / bar['close'], 1.0)
                })
            
            # Mean Reversion Strategy
            if bar['rsi'] < 30 and bar['close'] < bar['bb_lower']:
                signals.append({
                    'idx': i,
                    'strategy': 'MeanReversion',
                    'side': 'BUY',
                    'signal_strength': min((30 - bar['rsi']) / 30, 1.0)
                })
            elif bar['rsi'] > 70 and bar['close'] > bar['bb_upper']:
                signals.append({
                    'idx': i,
                    'strategy': 'MeanReversion',
                    'side': 'SELL', 
                    'signal_strength': min((bar['rsi'] - 70) / 30, 1.0)
                })
            
            # Breakout Strategy
            if bar['volume_ratio'] > 1.5 and bar['close'] > bar['bb_upper']:
                signals.append({
                    'idx': i,
                    'strategy': 'Breakout',
                    'side': 'BUY',
                    'signal_strength': min(bar['volume_ratio'] / 2.0, 1.0)
                })
            elif bar['volume_ratio'] > 1.5 and bar['close'] < bar['bb_lower']:
                signals.append({
                    'idx': i,
                    'strategy': 'Breakout',
                    'side': 'SELL',
                    'signal_strength': min(bar['volume_ratio'] / 2.0, 1.0)
                })
            
            # Momentum Strategy  
            if bar['price_momentum'] > 0.02 and bar['rsi'] > 60:
                signals.append({
                    'idx': i,
                    'strategy': 'Momentum',
                    'side': 'BUY',
                    'signal_strength': min(bar['price_momentum'] * 10, 1.0)
                })
            elif bar['price_momentum'] < -0.02 and bar['rsi'] < 40:
                signals.append({
                    'idx': i,
                    'strategy': 'Momentum',
                    'side': 'SELL',
                    'signal_strength': min(abs(bar['price_momentum']) * 10, 1.0)
                })
        
        logger.info(f"Generated {len(signals)} synthetic signals")
        return signals
    
    def create_training_candidates(self, df: pd.DataFrame, signals: List[Dict]) -> pd.DataFrame:
        """Create candidate training records matching real data format."""
        candidates = []
        rng = np.random.default_rng(42)  # Create random generator
        
        for signal in signals:
            try:
                idx = signal['idx']
                bar = df.iloc[idx]
                
                # Look ahead for outcome (triple barrier method approximation)
                entry_price = bar['close']
                stop_distance = bar['atr'] * 2.0  # 2 ATR stop
                target_distance = bar['atr'] * 3.0  # 3 ATR target (1.5R)
                
                if signal['side'] == 'BUY':
                    stop_price = entry_price - stop_distance
                    target_price = entry_price + target_distance
                else:
                    stop_price = entry_price + stop_distance  
                    target_price = entry_price - target_distance
                
                # Look ahead up to 100 bars for outcome
                outcome = self._determine_outcome(df, idx, entry_price, stop_price, target_price, signal['side'])
                
                # Session classification (approximate)
                hour = pd.to_datetime(bar['timestamp']).hour
                if 9 <= hour <= 16:
                    session = 'RTH'
                else:
                    session = 'ETH'
                
                # Create candidate record
                candidate = {
                    'timestamp': bar['timestamp'],
                    'symbol': 'ES',
                    'strategy': signal['strategy'],
                    'session': session,
                    'regime': bar['regime'],
                    'signal_id': f"vendor_{idx}_{signal['strategy'].lower()}",
                    
                    # Price features
                    'price': float(entry_price),
                    'atr': float(bar['atr']) if pd.notna(bar['atr']) else 1.0,
                    'rsi': float(bar['rsi']) if pd.notna(bar['rsi']) else 50.0,
                    'ema20': float(bar['ema20']) if pd.notna(bar['ema20']) else entry_price,
                    'ema50': float(bar['ema50']) if pd.notna(bar['ema50']) else entry_price,
                    
                    # Market microstructure  
                    'volume': float(bar['volume']) if pd.notna(bar['volume']) else 1000.0,
                    'spread': float(bar['spread_estimate']) if pd.notna(bar['spread_estimate']) else 1.0,
                    'volatility': float(bar['vol_rolling']) if pd.notna(bar['vol_rolling']) else 0.01,
                    'bid_ask_imbalance': float(rng.uniform(-0.1, 0.1)),  # Synthetic
                    'order_book_imbalance': float(rng.uniform(-0.1, 0.1)),  # Synthetic
                    'tick_direction': float(bar['tick_direction']) if pd.notna(bar['tick_direction']) else 0.0,
                    
                    # Strategy-specific
                    'signal_strength': float(signal['signal_strength']),
                    'prior_win_rate': 0.52,  # Approximate historical
                    'avg_r_multiple': 0.8,   # Approximate historical
                    
                    # Risk factors
                    'drawdown_risk': float(abs(bar['vol_zscore'])) if pd.notna(bar['vol_zscore']) else 0.0,
                    'news_impact': 0.0,  # No news data in vendor feed
                    'liquidity_risk': float(1.0 / bar['volume_ratio']) if pd.notna(bar['volume_ratio']) else 0.1,
                    
                    # Position sizing (what we're learning)
                    'baseline_multiplier': 1.0,
                    
                    # Outcome (labels)
                    'label_win': int(outcome['win']),
                    'r_multiple': float(outcome['r_multiple']),
                    'slip_ticks': float(outcome['slip_ticks'])
                }
                
                candidates.append(candidate)
                
            except Exception as ex:
                logger.warning(f"Failed to create candidate for signal {signal}: {ex}")
                continue
        
        candidate_df = pd.DataFrame(candidates)
        logger.info(f"Created {len(candidate_df)} training candidates")
        logger.info(f"Win rate: {candidate_df['label_win'].mean():.3f}")
        logger.info(f"Avg R-multiple: {candidate_df['r_multiple'].mean():.3f}")
        
        return candidate_df
    
    def _determine_outcome(self, df: pd.DataFrame, start_idx: int, entry: float, stop: float, target: float, side: str) -> Dict:
        """Determine trade outcome using triple barrier method."""
        outcome = {'win': 0, 'r_multiple': 0.0, 'slip_ticks': 0.1}
        rng = np.random.default_rng(42)  # Create random generator for this method
        
        try:
            max_bars = min(100, len(df) - start_idx - 1)
            if max_bars <= 0:
                return outcome
            
            for i in range(1, max_bars + 1):
                bar = df.iloc[start_idx + i]
                
                if side == 'BUY':
                    # Check stop hit
                    if bar['low'] <= stop:
                        outcome['win'] = 0
                        outcome['r_multiple'] = (stop - entry) / abs(entry - stop)  # Should be -1.0
                        outcome['slip_ticks'] = rng.exponential(0.5)
                        break
                    # Check target hit
                    elif bar['high'] >= target:
                        outcome['win'] = 1
                        outcome['r_multiple'] = (target - entry) / abs(entry - stop)  # Should be ~1.5
                        outcome['slip_ticks'] = rng.exponential(0.3)
                        break
                else:  # SELL
                    # Check stop hit
                    if bar['high'] >= stop:
                        outcome['win'] = 0
                        outcome['r_multiple'] = (entry - stop) / abs(entry - stop)  # Should be -1.0
                        outcome['slip_ticks'] = rng.exponential(0.5) 
                        break
                    # Check target hit
                    elif bar['low'] <= target:
                        outcome['win'] = 1
                        outcome['r_multiple'] = (entry - target) / abs(entry - stop)  # Should be ~1.5
                        outcome['slip_ticks'] = rng.exponential(0.3)
                        break
            
            # If no stop or target hit, use time-based exit
            if abs(outcome['r_multiple']) < 1e-6:  # Check for near-zero instead of exact equality
                final_bar = df.iloc[start_idx + max_bars]
                exit_price = final_bar['close']
                
                if side == 'BUY':
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price
                    
                risk = abs(entry - stop)
                outcome['r_multiple'] = pnl / risk if risk > 0 else 0.0
                outcome['win'] = 1 if pnl > 0 else 0
                outcome['slip_ticks'] = rng.exponential(0.4)
                
        except Exception as ex:
            logger.warning(f"Failed to determine outcome: {ex}")
            
        return outcome
    
    def generate(self, symbol: str = "ES") -> bool:
        """Main generation pipeline."""
        try:
            logger.info(f"🤖 Starting vendor feature generation for {symbol}")
            
            # Load vendor data
            df = self.load_vendor_data(symbol)
            if df is None or len(df) < self.min_bars:
                logger.warning(f"Insufficient vendor data for {symbol}")
                return False
            
            # Compute indicators
            df = self.compute_technical_indicators(df)
            
            # Identify regimes
            df = self.identify_market_regimes(df)
            
            # Generate signals
            signals = self.generate_synthetic_signals(df)
            if len(signals) == 0:  # Check length instead of boolean
                logger.warning("No signals generated")
                return False
            
            # Create training candidates
            candidates = self.create_training_candidates(df, signals)
            if candidates.empty:
                logger.warning("No candidates created")
                return False
            
            # Save to output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_parquet(self.output_path, index=False)
            
            logger.info(f"✅ Generated {len(candidates)} vendor training candidates")
            logger.info(f"📁 Saved to: {self.output_path}")
            logger.info(f"📊 Strategy distribution: {candidates['strategy'].value_counts().to_dict()}")
            
            return True
            
        except Exception as ex:
            logger.error(f"Vendor feature generation failed: {ex}")
            return False


def main():
    """CLI entry point."""
    vendor_dir = os.getenv('VENDOR_DIR', 'data/vendor')
    output_path = os.getenv('OUT_PATH', 'data/logs/candidates.vendor.parquet')
    symbol = os.getenv('SYMBOL', 'ES')
    
    generator = VendorFeatureGenerator(vendor_dir, output_path)
    success = generator.generate(symbol)
    
    if success:
        logger.info("🎯 Vendor feature generation completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Vendor feature generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
