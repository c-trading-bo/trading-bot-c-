#!/usr/bin/env python3
"""
🎯 Complete S3 Strategy Historical Backtester
Shows how S3 compression/breakout strategy performs on real ES/NQ market data
Uses built-in technical indicators - no external dependencies needed
"""
import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Tuple

class TechnicalIndicators:
    """Built-in technical indicators for S3 strategy"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(0.0)
            else:
                avg = sum(prices[i - period + 1:i + 1]) / period
                result.append(avg)
        return result
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if not prices:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema_values = [prices[0]]  # Start with first price
        
        for i in range(1, len(prices)):
            ema_val = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema_val)
        
        return ema_values
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
        """Average True Range"""
        if len(highs) != len(lows) or len(lows) != len(closes):
            raise ValueError("All price arrays must be same length")
        
        true_ranges = []
        for i in range(len(highs)):
            if i == 0:
                tr = highs[i] - lows[i]
            else:
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr = max(tr1, tr2, tr3)
            true_ranges.append(tr)
        
        # Calculate ATR using SMA of true ranges
        return TechnicalIndicators.sma(true_ranges, period)
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int, std_dev: float) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        sma_values = TechnicalIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(0.0)
                lower_band.append(0.0)
            else:
                # Calculate standard deviation for this window
                window = prices[i - period + 1:i + 1]
                mean = sma_values[i]
                variance = sum((x - mean) ** 2 for x in window) / period
                std = math.sqrt(variance)
                
                upper_band.append(mean + (std * std_dev))
                lower_band.append(mean - (std * std_dev))
        
        return upper_band, sma_values, lower_band


class S3StrategyBacktester:
    """Complete S3 Strategy Backtester"""
    
    def __init__(self):
        # S3 Strategy Parameters
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.keltner_period = 20
        self.keltner_multiplier = 1.5
        self.volume_threshold = 1.2
        
        # Trading time windows (EST)
        self.trading_windows = [
            (time(9, 40), time(10, 30)),   # Morning session
            (time(14, 55), time(16, 10))   # Afternoon session
        ]
        
        # Point values for P&L calculation
        self.point_values = {
            'ES': 50.0,  # $50 per point
            'NQ': 20.0   # $20 per point
        }
    
    def load_historical_data(self, symbol: str) -> List[Dict]:
        """Load real historical data for symbol"""
        
        data_file = Path(f"data/historical/{symbol}_bars.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"❌ Historical data not found: {data_file}")
        
        print(f"📊 Loading real {symbol} historical data...")
        
        with open(data_file, 'r') as f:
            bars = json.load(f)
        
        print(f"   ✅ Loaded {len(bars)} real {symbol} bars")
        
        if bars:
            start_date = datetime.fromisoformat(bars[0]['Start'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(bars[-1]['Start'].replace('Z', '+00:00'))
            print(f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return bars
    
    def calculate_indicators(self, bars: List[Dict]) -> Dict:
        """Calculate all technical indicators for S3 strategy"""
        
        print(f"🔧 Calculating technical indicators...")
        
        # Extract price arrays
        opens = [float(bar['Open']) for bar in bars]
        highs = [float(bar['High']) for bar in bars]
        lows = [float(bar['Low']) for bar in bars]
        closes = [float(bar['Close']) for bar in bars]
        volumes = [float(bar['Volume']) for bar in bars]
        
        # Calculate indicators
        indicators = {}
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            closes, self.bollinger_period, self.bollinger_std
        )
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Keltner Channels
        atr_values = TechnicalIndicators.atr(highs, lows, closes, self.keltner_period)
        kc_middle = TechnicalIndicators.ema(closes, self.keltner_period)
        kc_upper = [mid + (atr * self.keltner_multiplier) if atr > 0 else 0 
                   for mid, atr in zip(kc_middle, atr_values)]
        kc_lower = [mid - (atr * self.keltner_multiplier) if atr > 0 else 0 
                   for mid, atr in zip(kc_middle, atr_values)]
        
        indicators['kc_upper'] = kc_upper
        indicators['kc_middle'] = kc_middle
        indicators['kc_lower'] = kc_lower
        indicators['atr'] = atr_values
        
        # Volume indicators
        volume_sma = TechnicalIndicators.sma(volumes, 20)
        volume_ratio = [vol / sma if sma > 0 else 0 for vol, sma in zip(volumes, volume_sma)]
        indicators['volume_ratio'] = volume_ratio
        
        # Compression detection (BB inside KC)
        compression = []
        for i in range(len(bars)):
            if (i >= self.bollinger_period and 
                bb_upper[i] > 0 and bb_lower[i] > 0 and 
                kc_upper[i] > 0 and kc_lower[i] > 0):
                is_compressed = (bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i])
                compression.append(is_compressed)
            else:
                compression.append(False)
        
        indicators['compression'] = compression
        
        print(f"   ✅ Calculated {len(indicators)} indicator sets")
        
        return indicators
    
    def is_trading_time(self, timestamp_str: str) -> bool:
        """Check if timestamp is within trading windows"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = dt.time()
            
            for start_time, end_time in self.trading_windows:
                if start_time <= current_time <= end_time:
                    return True
            return False
        except:
            return False
    
    def generate_signals(self, bars: List[Dict], indicators: Dict, symbol: str) -> List[Dict]:
        """Generate S3 trading signals"""
        
        print(f"📡 Generating S3 trading signals...")
        
        signals = []
        
        for i in range(self.bollinger_period, len(bars) - 5):  # Need lookback and lookahead
            bar = bars[i]
            
            # Only trade during specified windows
            if not self.is_trading_time(bar['Start']):
                continue
            
            # Check for compression
            if not indicators['compression'][i]:
                continue
            
            # Look for breakout in next few bars
            for lookhead in range(1, 4):
                idx = i + lookhead
                if idx >= len(bars):
                    break
                
                current_bar = bars[idx]
                current_close = float(current_bar['Close'])
                current_volume_ratio = indicators['volume_ratio'][idx]
                
                # Bullish breakout: Close above KC upper with volume
                if (current_close > indicators['kc_upper'][idx] and
                    current_volume_ratio > self.volume_threshold and
                    indicators['kc_upper'][idx] > 0):
                    
                    signal = {
                        'symbol': symbol,
                        'timestamp': current_bar['Start'],
                        'signal': 'BUY',
                        'price': current_close,
                        'reason': 'Compression_Breakout_Bull',
                        'atr': indicators['atr'][idx],
                        'volume_ratio': current_volume_ratio,
                        'compression_bars': lookhead,
                        'bb_width': indicators['bb_upper'][i] - indicators['bb_lower'][i],
                        'kc_width': indicators['kc_upper'][i] - indicators['kc_lower'][i]
                    }
                    signals.append(signal)
                    break
                
                # Bearish breakout: Close below KC lower with volume
                elif (current_close < indicators['kc_lower'][idx] and
                      current_volume_ratio > self.volume_threshold and
                      indicators['kc_lower'][idx] > 0):
                    
                    signal = {
                        'symbol': symbol,
                        'timestamp': current_bar['Start'],
                        'signal': 'SELL',
                        'price': current_close,
                        'reason': 'Compression_Breakout_Bear',
                        'atr': indicators['atr'][idx],
                        'volume_ratio': current_volume_ratio,
                        'compression_bars': lookhead,
                        'bb_width': indicators['bb_upper'][i] - indicators['bb_lower'][i],
                        'kc_width': indicators['kc_upper'][i] - indicators['kc_lower'][i]
                    }
                    signals.append(signal)
                    break
        
        print(f"   ✅ Generated {len(signals)} trading signals")
        
        return signals
    
    def calculate_performance(self, signals: List[Dict], bars: List[Dict], symbol: str) -> Dict:
        """Calculate strategy performance metrics"""
        
        if not signals:
            return {
                'total_signals': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'trade_details': []
            }
        
        print(f"📊 Calculating performance for {len(signals)} signals...")
        
        point_value = self.point_values.get(symbol, 50.0)
        
        # Create price lookup for faster access
        price_lookup = {}
        for i, bar in enumerate(bars):
            timestamp = bar['Start']
            price_lookup[timestamp] = (i, float(bar['Close']))
        
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        trade_pnls = []
        trade_details = []
        
        for signal in signals:
            entry_time = signal['timestamp']
            entry_price = signal['price']
            
            # Find entry index
            if entry_time not in price_lookup:
                continue
            
            entry_idx, _ = price_lookup[entry_time]
            
            # Hold for 5 bars (simple exit strategy)
            exit_idx = entry_idx + 5
            if exit_idx >= len(bars):
                continue
            
            exit_price = float(bars[exit_idx]['Close'])
            exit_time = bars[exit_idx]['Start']
            
            # Calculate P&L
            if signal['signal'] == 'BUY':
                pnl = (exit_price - entry_price) * point_value
            else:  # SELL
                pnl = (entry_price - exit_price) * point_value
            
            total_pnl += pnl
            trade_pnls.append(pnl)
            
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            # Store trade details
            trade_details.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'signal': signal['signal'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': signal['reason'],
                'atr': signal['atr'],
                'volume_ratio': signal['volume_ratio']
            })
        
        return {
            'total_signals': len(signals),
            'total_pnl': total_pnl,
            'win_rate': (winning_trades / len(trade_pnls)) * 100 if trade_pnls else 0,
            'avg_trade': total_pnl / len(trade_pnls) if trade_pnls else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'max_win': max(trade_pnls) if trade_pnls else 0,
            'max_loss': min(trade_pnls) if trade_pnls else 0,
            'trade_details': trade_details[:10]  # Show first 10 trades
        }
    
    def run_backtest(self, symbol: str) -> Dict:
        """Run complete backtest on real historical data"""
        
        print(f"\n🎯 Running S3 Strategy Backtest on Real {symbol} Data")
        print("=" * 60)
        
        # Load real market data
        bars = self.load_historical_data(symbol)
        
        # Calculate technical indicators
        indicators = self.calculate_indicators(bars)
        
        # Generate trading signals
        signals = self.generate_signals(bars, indicators, symbol)
        
        # Calculate performance
        performance = self.calculate_performance(signals, bars, symbol)
        
        return {
            'symbol': symbol,
            'data_bars': len(bars),
            'signals': signals,
            'performance': performance,
            'indicators_calculated': len(indicators)
        }
    
    def display_results(self, results: Dict):
        """Display comprehensive backtest results"""
        
        symbol = results['symbol']
        perf = results['performance']
        
        print(f"\n📈 {symbol} S3 Strategy Performance Results")
        print("=" * 50)
        print(f"📊 Total Historical Bars: {results['data_bars']:,}")
        print(f"🎯 Total Trading Signals: {perf['total_signals']}")
        print(f"💰 Total P&L: ${perf['total_pnl']:,.2f}")
        print(f"🎲 Win Rate: {perf['win_rate']:.1f}%")
        print(f"📊 Average Trade: ${perf['avg_trade']:,.2f}")
        print(f"✅ Winning Trades: {perf['winning_trades']}")
        print(f"❌ Losing Trades: {perf['losing_trades']}")
        print(f"🎉 Best Trade: ${perf['max_win']:,.2f}")
        print(f"😬 Worst Trade: ${perf['max_loss']:,.2f}")
        
        # Show sample trades
        if perf['trade_details']:
            print(f"\n🔍 Sample Trading Activity:")
            for i, trade in enumerate(perf['trade_details'][:5]):
                dt = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                pnl_color = "💚" if trade['pnl'] > 0 else "❤️"
                print(f"   {i+1}. {dt.strftime('%m/%d %H:%M')} - "
                      f"{trade['signal']} {symbol} at ${trade['entry_price']:.2f} → "
                      f"${trade['exit_price']:.2f} = {pnl_color}${trade['pnl']:,.2f}")
        
        print()


def main():
    """Run complete historical backtesting setup"""
    
    print("🚀 Complete S3 Strategy Historical Backtester")
    print("📈 Testing on REAL ES and NQ historical market data")
    print("🎯 Data Flow: Real OHLC → Technical Indicators → Compression Detection → Trading Signals → P&L")
    print("=" * 80)
    
    backtester = S3StrategyBacktester()
    
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
    
    # Final Summary
    if all_results:
        print("\n🏆 FINAL S3 STRATEGY SUMMARY")
        print("=" * 50)
        
        total_signals = sum(r['performance']['total_signals'] for r in all_results.values())
        total_pnl = sum(r['performance']['total_pnl'] for r in all_results.values())
        total_bars = sum(r['data_bars'] for r in all_results.values())
        
        print(f"📊 Combined Signals: {total_signals}")
        print(f"💰 Combined P&L: ${total_pnl:,.2f}")
        print(f"🎯 Total Real Market Bars Analyzed: {total_bars:,}")
        print(f"📅 Real market data from past 3 weeks")
        
        if total_pnl > 0:
            print("✅ S3 Strategy shows POSITIVE performance on real historical data!")
            print("🎉 Your compression/breakout strategy would have been profitable!")
        else:
            print("⚠️  S3 Strategy shows challenges on this period")
            print("🔧 Consider adjusting parameters or different time windows")
        
        print(f"\n🎯 S3 Strategy Characteristics Tested:")
        print(f"   • Bollinger Bands (20 period, 2.0 std)")
        print(f"   • Keltner Channels (20 period, 1.5 multiplier)")
        print(f"   • Volume threshold: 1.2x average")
        print(f"   • Trading windows: 09:40-10:30, 14:55-16:10")
        print(f"   • Real ES/NQ market moves")


if __name__ == "__main__":
    main()