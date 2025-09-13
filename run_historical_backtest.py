#!/usr/bin/env python3
"""
🎯 Real Historical Data Backtester for S3 Strategy
Shows how S3 compression/breakout strategy performs on actual ES/NQ market data
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Tuple
import talib

class S3Strategy:
    """S3 Compression/Breakout Strategy Implementation"""
    
    def __init__(self):
        # S3 Strategy Parameters (from your config)
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.keltner_period = 20
        self.keltner_multiplier = 1.5
        self.volume_threshold = 1.2
        
        # Trading time windows (from your S3 config)
        self.trading_windows = [
            (time(9, 40), time(10, 30)),   # Morning session
            (time(14, 55), time(16, 10))   # Afternoon session (02:55-04:10 PM)
        ]
        
        self.signals = []
        self.trades = []
        self.positions = {}
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for S3 strategy"""
        
        # Convert OHLC to numpy arrays for TA-Lib
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        volume = df['Volume'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                     timeperiod=self.bollinger_period, 
                                                     nbdevup=self.bollinger_std,
                                                     nbdevdn=self.bollinger_std)
        
        # Keltner Channels (using ATR)
        atr = talib.ATR(high, low, close, timeperiod=self.keltner_period)
        ema = talib.EMA(close, timeperiod=self.keltner_period)
        keltner_upper = ema + (atr * self.keltner_multiplier)
        keltner_lower = ema - (atr * self.keltner_multiplier)
        
        # Volume indicators
        volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
        
        # Add to dataframe
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['KC_Upper'] = keltner_upper
        df['KC_Lower'] = keltner_lower
        df['ATR'] = atr
        df['EMA_20'] = ema
        df['Volume_SMA'] = volume_sma
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Compression detection (BB inside KC)
        df['Compression'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
        
        return df
    
    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading windows"""
        current_time = timestamp.time()
        for start_time, end_time in self.trading_windows:
            if start_time <= current_time <= end_time:
                return True
        return False
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate S3 trading signals"""
        signals = []
        
        for i in range(len(df)):
            if i < self.bollinger_period:  # Not enough data
                continue
                
            row = df.iloc[i]
            timestamp = pd.to_datetime(row['Start'])
            
            # Only trade during specified windows
            if not self.is_trading_time(timestamp):
                continue
            
            # Check for compression breakout
            if pd.notna(row['Compression']) and row['Compression']:
                # Look for breakout in next few bars
                breakout_signal = self.detect_breakout(df, i, symbol)
                if breakout_signal:
                    signals.append(breakout_signal)
        
        return signals
    
    def detect_breakout(self, df: pd.DataFrame, compression_idx: int, symbol: str) -> Dict:
        """Detect breakout from compression"""
        
        # Look ahead 1-3 bars for breakout
        for lookhead in range(1, min(4, len(df) - compression_idx)):
            idx = compression_idx + lookhead
            if idx >= len(df):
                break
                
            current = df.iloc[idx]
            prev = df.iloc[idx - 1]
            
            # Bullish breakout: Close above KC upper
            if (current['Close'] > current['KC_Upper'] and 
                prev['Close'] <= prev['KC_Upper'] and
                current['Volume_Ratio'] > self.volume_threshold):
                
                return {
                    'symbol': symbol,
                    'timestamp': pd.to_datetime(current['Start']),
                    'signal': 'BUY',
                    'price': current['Close'],
                    'reason': 'Compression_Breakout_Bull',
                    'atr': current['ATR'],
                    'volume_ratio': current['Volume_Ratio'],
                    'compression_bars': lookhead
                }
            
            # Bearish breakout: Close below KC lower  
            elif (current['Close'] < current['KC_Lower'] and 
                  prev['Close'] >= prev['KC_Lower'] and
                  current['Volume_Ratio'] > self.volume_threshold):
                
                return {
                    'symbol': symbol,
                    'timestamp': pd.to_datetime(current['Start']),
                    'signal': 'SELL',
                    'price': current['Close'],
                    'reason': 'Compression_Breakout_Bear',
                    'atr': current['ATR'],
                    'volume_ratio': current['Volume_Ratio'],
                    'compression_bars': lookhead
                }
        
        return None


class HistoricalBacktester:
    """Real historical data backtester"""
    
    def __init__(self):
        self.strategy = S3Strategy()
        self.results = {}
        
    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """Load real historical data for symbol"""
        
        data_file = Path(f"data/historical/{symbol}_bars.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"Historical data not found: {data_file}")
        
        print(f"📊 Loading real {symbol} historical data...")
        
        with open(data_file, 'r') as f:
            bars = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df['Start'] = pd.to_datetime(df['Start'])
        df = df.sort_values('Start').reset_index(drop=True)
        
        print(f"   ✅ Loaded {len(df)} real {symbol} bars")
        print(f"   📅 Date range: {df['Start'].min()} to {df['Start'].max()}")
        
        return df
    
    def run_backtest(self, symbol: str) -> Dict:
        """Run complete backtest on real historical data"""
        
        print(f"\n🎯 Running S3 Strategy Backtest on Real {symbol} Data")
        print("=" * 60)
        
        # Load real market data
        df = self.load_historical_data(symbol)
        
        # Calculate technical indicators
        print(f"🔧 Calculating technical indicators...")
        df = self.strategy.calculate_indicators(df)
        
        # Generate trading signals
        print(f"📡 Generating S3 trading signals...")
        signals = self.strategy.generate_signals(df, symbol)
        
        # Calculate performance
        performance = self.calculate_performance(signals, df, symbol)
        
        return {
            'symbol': symbol,
            'data_bars': len(df),
            'date_range': (df['Start'].min(), df['Start'].max()),
            'signals': signals,
            'performance': performance,
            'data_sample': df.head(5).to_dict('records')
        }
    
    def calculate_performance(self, signals: List[Dict], df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate strategy performance metrics"""
        
        if not signals:
            return {
                'total_signals': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_trade': 0.0
            }
        
        total_pnl = 0.0
        winning_trades = 0
        trade_pnls = []
        
        # Simple P&L calculation (assuming $50/point for ES, $20/point for NQ)
        point_value = 50.0 if symbol == 'ES' else 20.0
        
        for i, signal in enumerate(signals):
            # Simulate holding for 5 bars (basic exit strategy)
            entry_price = signal['price']
            
            # Find exit (5 bars later or end of data)
            entry_time = signal['timestamp']
            exit_idx = None
            
            for j, row in df.iterrows():
                if pd.to_datetime(row['Start']) > entry_time:
                    if exit_idx is None:
                        exit_idx = j
                    elif j >= exit_idx + 5:  # Hold for 5 bars
                        break
            
            if exit_idx and exit_idx + 5 < len(df):
                exit_price = df.iloc[exit_idx + 5]['Close']
                
                if signal['signal'] == 'BUY':
                    pnl = (exit_price - entry_price) * point_value
                else:  # SELL
                    pnl = (entry_price - exit_price) * point_value
                
                total_pnl += pnl
                trade_pnls.append(pnl)
                
                if pnl > 0:
                    winning_trades += 1
        
        return {
            'total_signals': len(signals),
            'total_pnl': total_pnl,
            'win_rate': (winning_trades / len(trade_pnls)) * 100 if trade_pnls else 0,
            'avg_trade': total_pnl / len(trade_pnls) if trade_pnls else 0,
            'winning_trades': winning_trades,
            'losing_trades': len(trade_pnls) - winning_trades,
            'trade_pnls': trade_pnls[:10]  # Show first 10 trades
        }
    
    def display_results(self, results: Dict):
        """Display backtest results"""
        
        symbol = results['symbol']
        perf = results['performance']
        signals = results['signals']
        
        print(f"\n📈 {symbol} Strategy Performance Results")
        print("=" * 50)
        print(f"📊 Data Period: {results['date_range'][0].strftime('%Y-%m-%d')} to {results['date_range'][1].strftime('%Y-%m-%d')}")
        print(f"📋 Total Bars: {results['data_bars']:,}")
        print(f"🎯 Total Signals: {perf['total_signals']}")
        print(f"💰 Total P&L: ${perf['total_pnl']:,.2f}")
        print(f"🎲 Win Rate: {perf['win_rate']:.1f}%")
        print(f"📊 Avg Trade: ${perf['avg_trade']:,.2f}")
        print(f"✅ Winning Trades: {perf['winning_trades']}")
        print(f"❌ Losing Trades: {perf['losing_trades']}")
        
        # Show first few signals
        if signals:
            print(f"\n🔍 First 5 Trading Signals:")
            for i, signal in enumerate(signals[:5]):
                print(f"   {i+1}. {signal['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                      f"{signal['signal']} {symbol} at ${signal['price']:.2f} "
                      f"({signal['reason']}, Vol: {signal['volume_ratio']:.1f}x)")
        
        print()


def main():
    """Run historical backtesting on real ES/NQ data"""
    
    print("🚀 S3 Strategy Historical Backtester")
    print("📈 Testing on REAL ES and NQ historical market data")
    print("🎯 Data Flow: OHLC → Technical Indicators → Compression Detection → Trading Signals")
    print("=" * 80)
    
    backtester = HistoricalBacktester()
    
    # Test both ES and NQ
    symbols = ['ES', 'NQ']
    all_results = {}
    
    for symbol in symbols:
        try:
            results = backtester.run_backtest(symbol)
            all_results[symbol] = results
            backtester.display_results(results)
            
        except FileNotFoundError as e:
            print(f"❌ {symbol}: {e}")
        except Exception as e:
            print(f"💥 {symbol}: Error - {e}")
    
    # Summary
    if all_results:
        print("\n🏆 FINAL SUMMARY")
        print("=" * 50)
        total_signals = sum(r['performance']['total_signals'] for r in all_results.values())
        total_pnl = sum(r['performance']['total_pnl'] for r in all_results.values())
        
        print(f"📊 Total Signals (All Symbols): {total_signals}")
        print(f"💰 Combined P&L: ${total_pnl:,.2f}")
        print(f"🎯 Strategy tested on {sum(r['data_bars'] for r in all_results.values()):,} real market bars")
        print(f"📅 Covering real market moves from past 3 weeks")
        
        if total_pnl > 0:
            print("✅ S3 Strategy shows POSITIVE performance on real historical data!")
        else:
            print("⚠️  S3 Strategy shows challenges - consider parameter tuning")


if __name__ == "__main__":
    main()