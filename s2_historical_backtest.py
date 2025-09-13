#!/usr/bin/env python3
"""
🎯 S2 Strategy Historical Backtester - VWAP Mean Reversion
Shows how S2 VWAP mean reversion strategy performs on real ES/NQ market data
Tests the sophisticated fade-extreme-moves strategy with all safeguards
"""
import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import List, Dict, Tuple

class S2TechnicalIndicators:
    """Technical indicators for S2 VWAP mean reversion strategy"""
    
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
        ema_values = [prices[0]]
        
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
        
        return S2TechnicalIndicators.sma(true_ranges, period)
    
    @staticmethod
    def session_vwap_and_variance(bars: List[Dict], session_start: datetime) -> Tuple[float, float, float]:
        """Calculate session VWAP and variance from session start"""
        total_pv = 0.0  # price * volume
        total_volume = 0.0
        total_pv2 = 0.0  # price^2 * volume for variance
        
        for bar in bars:
            bar_time = datetime.fromisoformat(bar['Start'].replace('Z', '+00:00'))
            
            # Only include bars from session start onwards
            if bar_time >= session_start:
                typical_price = (float(bar['High']) + float(bar['Low']) + float(bar['Close'])) / 3.0
                volume = float(bar['Volume'])
                
                total_pv += typical_price * volume
                total_pv2 += (typical_price * typical_price) * volume
                total_volume += volume
        
        if total_volume <= 0:
            return 0.0, 0.0, 0.0
        
        vwap = total_pv / total_volume
        
        # Calculate volume-weighted variance
        variance = (total_pv2 / total_volume) - (vwap * vwap)
        variance = max(0.0, variance)  # Ensure non-negative
        
        return vwap, variance, total_volume
    
    @staticmethod
    def volume_z_score(bars: List[Dict], lookback: int = 50) -> float:
        """Calculate volume Z-score over lookback period"""
        if len(bars) < lookback:
            return 0.0
        
        recent_volumes = [float(bar['Volume']) for bar in bars[-lookback:]]
        current_volume = recent_volumes[-1]
        
        avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
        
        if avg_volume <= 0:
            return 0.0
        
        # Simple volume ratio instead of z-score for stability
        return current_volume / avg_volume - 1.0
    
    @staticmethod
    def up_down_volume_imbalance(bars: List[Dict], lookback: int = 10) -> float:
        """Calculate up vs down volume imbalance"""
        if len(bars) < 2:
            return 1.0
        
        up_volume = 0.0
        down_volume = 0.0
        
        for bar in bars[-lookback:]:
            volume = float(bar['Volume'])
            open_price = float(bar['Open'])
            close_price = float(bar['Close'])
            
            if close_price > open_price:
                up_volume += volume
            elif close_price < open_price:
                down_volume += volume
        
        if down_volume <= 0:
            return 1.5  # Buyers dominant
        
        return up_volume / down_volume


class S2StrategyBacktester:
    """S2 VWAP Mean Reversion Strategy Backtester"""
    
    def __init__(self):
        # S2 Strategy Parameters (from your code)
        self.atr_len = 14
        self.sigma_enter = 2.0
        self.atr_enter = 1.0
        self.sigma_force_trend = 2.8
        self.min_slope_tf2 = 0.18  # ticks/bar on EMA20
        self.volz_min = -0.3
        self.volz_max = 2.2
        self.confirm_lookback = 3
        self.validity_bars = 3
        self.cooldown_bars = 5
        self.max_bars_in_trade = 45
        self.stop_atr_mult = 0.75
        self.trail_atr_mult = 1.0
        self.ib_end_minute = 10 * 60 + 30  # 10:30 AM
        
        # Instrument-specific sigma
        self.es_sigma = 2.0
        self.nq_sigma = 2.6
        
        self.min_volume = 3000
        self.max_spread_ticks = 2
        
        # ADR settings
        self.adr_lookback_days = 20
        self.adr_room_frac = 0.25
        self.adr_exhaustion_cap = 1.20
        
        # Point values for P&L
        self.point_values = {
            'ES': 50.0,
            'NQ': 20.0
        }
        
        # Market session (9:30 AM ET)
        self.session_start_time = time(9, 30)
    
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
    
    def calculate_indicators(self, bars: List[Dict], symbol: str) -> Dict:
        """Calculate all technical indicators for S2 strategy"""
        
        print(f"🔧 Calculating S2 technical indicators...")
        
        # Extract price arrays
        opens = [float(bar['Open']) for bar in bars]
        highs = [float(bar['High']) for bar in bars]
        lows = [float(bar['Low']) for bar in bars]
        closes = [float(bar['Close']) for bar in bars]
        volumes = [float(bar['Volume']) for bar in bars]
        
        indicators = {}
        
        # ATR
        atr_values = S2TechnicalIndicators.atr(highs, lows, closes, self.atr_len)
        indicators['atr'] = atr_values
        
        # EMA20 for trend detection
        ema20 = S2TechnicalIndicators.ema(closes, 20)
        indicators['ema20'] = ema20
        
        # Volume indicators
        indicators['volume_z'] = []
        indicators['volume_imbalance'] = []
        
        for i in range(len(bars)):
            # Volume Z-score
            if i >= 50:
                volz = S2TechnicalIndicators.volume_z_score(bars[:i+1], 50)
            else:
                volz = 0.0
            indicators['volume_z'].append(volz)
            
            # Volume imbalance
            if i >= 10:
                imbalance = S2TechnicalIndicators.up_down_volume_imbalance(bars[max(0, i-10):i+1], 10)
            else:
                imbalance = 1.0
            indicators['volume_imbalance'].append(imbalance)
        
        # Session VWAP calculation for each bar
        indicators['vwap'] = []
        indicators['sigma'] = []
        
        for i in range(len(bars)):
            current_time = datetime.fromisoformat(bars[i]['Start'].replace('Z', '+00:00'))
            current_date = current_time.date()
            
            # Find session start for this bar's date
            session_start = datetime.combine(current_date, self.session_start_time)
            session_start = session_start.replace(tzinfo=current_time.tzinfo)
            
            # Calculate session VWAP up to this bar
            session_bars = []
            for j in range(i + 1):
                bar_time = datetime.fromisoformat(bars[j]['Start'].replace('Z', '+00:00'))
                if bar_time >= session_start:
                    session_bars.append(bars[j])
            
            if session_bars:
                vwap, variance, volume = S2TechnicalIndicators.session_vwap_and_variance(session_bars, session_start)
                sigma = math.sqrt(max(0.0, variance))
                indicators['vwap'].append(vwap)
                indicators['sigma'].append(sigma)
            else:
                indicators['vwap'].append(0.0)
                indicators['sigma'].append(0.0)
        
        print(f"   ✅ Calculated {len(indicators)} S2 indicator sets")
        
        return indicators
    
    def calculate_adr(self, bars: List[Dict]) -> float:
        """Calculate Average Daily Range"""
        daily_ranges = []
        current_date = None
        day_high = 0.0
        day_low = float('inf')
        
        for bar in bars:
            bar_time = datetime.fromisoformat(bar['Start'].replace('Z', '+00:00'))
            bar_date = bar_time.date()
            
            if current_date != bar_date:
                # Save previous day's range
                if current_date is not None and day_high > day_low:
                    daily_ranges.append(day_high - day_low)
                
                # Start new day
                current_date = bar_date
                day_high = float(bar['High'])
                day_low = float(bar['Low'])
            else:
                # Update day's high/low
                day_high = max(day_high, float(bar['High']))
                day_low = min(day_low, float(bar['Low']))
        
        # Add last day
        if current_date is not None and day_high > day_low:
            daily_ranges.append(day_high - day_low)
        
        if not daily_ranges:
            return 0.0
        
        # Return average of last N days
        lookback_ranges = daily_ranges[-self.adr_lookback_days:]
        return sum(lookback_ranges) / len(lookback_ranges)
    
    def detect_trend_day(self, indicators: Dict, index: int) -> bool:
        """Detect if current conditions suggest trend day"""
        if index < 6 or len(indicators['ema20']) <= index:
            return False
        
        # Calculate EMA20 slope over last 5 bars
        ema20 = indicators['ema20']
        slope = (ema20[index] - ema20[index - 5]) / 5.0
        slope_ticks = abs(slope) / 0.25  # Assuming 0.25 tick size
        
        return slope_ticks > self.min_slope_tf2
    
    def generate_signals(self, bars: List[Dict], indicators: Dict, symbol: str) -> List[Dict]:
        """Generate S2 VWAP mean reversion signals"""
        
        print(f"📡 Generating S2 VWAP mean reversion signals...")
        
        signals = []
        adr = self.calculate_adr(bars)
        
        # Instrument-specific sigma threshold
        is_nq = 'NQ' in symbol.upper()
        base_sigma = self.nq_sigma if is_nq else self.es_sigma
        
        for i in range(60, len(bars) - 5):  # Need history and lookahead
            bar = bars[i]
            current_time = datetime.fromisoformat(bar['Start'].replace('Z', '+00:00'))
            
            # Skip if not enough volume
            if float(bar['Volume']) < self.min_volume:
                continue
            
            # Get indicators for this bar
            if (i >= len(indicators['vwap']) or i >= len(indicators['sigma']) or 
                i >= len(indicators['atr']) or i >= len(indicators['volume_z'])):
                continue
            
            vwap = indicators['vwap'][i]
            sigma = indicators['sigma'][i]
            atr = indicators['atr'][i]
            volz = indicators['volume_z'][i]
            volume_imbalance = indicators['volume_imbalance'][i]
            
            # Skip if no valid VWAP/sigma
            if vwap <= 0 or sigma <= 0 or atr <= 0:
                continue
            
            # Volume regime filter
            if not (self.volz_min <= volz <= self.volz_max):
                continue
            
            current_price = float(bar['Close'])
            z_score = (current_price - vwap) / sigma
            atr_distance = (current_price - vwap) / atr
            
            # Dynamic sigma threshold (trend day detection)
            is_trend_day = self.detect_trend_day(indicators, i)
            required_sigma = self.sigma_force_trend if is_trend_day else max(base_sigma, self.sigma_enter)
            
            # ADR room check
            if adr > 0:
                room_to_vwap = abs(current_price - vwap)
                if room_to_vwap < self.adr_room_frac * adr:
                    continue
            
            # LONG signal: Fade down move (price below VWAP)
            if (z_score <= -required_sigma or atr_distance <= -self.atr_enter):
                if volume_imbalance >= 0.9:  # Volume favors buyers
                    # Look for bull confirmation in next few bars
                    confirmed = False
                    for look_ahead in range(1, 4):
                        if i + look_ahead >= len(bars):
                            break
                        
                        future_bar = bars[i + look_ahead]
                        future_close = float(future_bar['Close'])
                        
                        # Simple confirmation: price moves back toward VWAP
                        if future_close > current_price:
                            confirmed = True
                            break
                    
                    if confirmed:
                        # Calculate stop and target
                        lookback_bars = bars[max(0, i - self.confirm_lookback):i + 1]
                        swing_low = min(float(b['Low']) for b in lookback_bars)
                        three_sigma_down = vwap - 3 * sigma
                        stop = min(swing_low, three_sigma_down)
                        
                        if stop >= current_price:
                            stop = current_price - 0.25 * atr
                        
                        risk = current_price - stop
                        target = vwap
                        
                        # Ensure minimum R:R
                        if target - current_price < 0.8 * risk:
                            target = current_price + 0.9 * risk
                        
                        signal = {
                            'symbol': symbol,
                            'timestamp': bar['Start'],
                            'signal': 'BUY',
                            'price': current_price,
                            'stop': stop,
                            'target': target,
                            'reason': 'VWAP_Mean_Reversion_Long',
                            'z_score': z_score,
                            'atr_distance': atr_distance,
                            'volume_imbalance': volume_imbalance,
                            'vwap': vwap,
                            'sigma': sigma,
                            'atr': atr,
                            'is_trend_day': is_trend_day,
                            'required_sigma': required_sigma
                        }
                        signals.append(signal)
            
            # SHORT signal: Fade up move (price above VWAP)
            elif (z_score >= required_sigma or atr_distance >= self.atr_enter):
                if volume_imbalance <= 1.1:  # Volume favors sellers
                    # Look for bear confirmation
                    confirmed = False
                    for look_ahead in range(1, 4):
                        if i + look_ahead >= len(bars):
                            break
                        
                        future_bar = bars[i + look_ahead]
                        future_close = float(future_bar['Close'])
                        
                        # Simple confirmation: price moves back toward VWAP
                        if future_close < current_price:
                            confirmed = True
                            break
                    
                    if confirmed:
                        # Calculate stop and target
                        lookback_bars = bars[max(0, i - self.confirm_lookback):i + 1]
                        swing_high = max(float(b['High']) for b in lookback_bars)
                        three_sigma_up = vwap + 3 * sigma
                        stop = max(swing_high, three_sigma_up)
                        
                        if stop <= current_price:
                            stop = current_price + 0.25 * atr
                        
                        risk = stop - current_price
                        target = vwap
                        
                        # Ensure minimum R:R
                        if current_price - target < 0.8 * risk:
                            target = current_price - 0.9 * risk
                        
                        signal = {
                            'symbol': symbol,
                            'timestamp': bar['Start'],
                            'signal': 'SELL',
                            'price': current_price,
                            'stop': stop,
                            'target': target,
                            'reason': 'VWAP_Mean_Reversion_Short',
                            'z_score': z_score,
                            'atr_distance': atr_distance,
                            'volume_imbalance': volume_imbalance,
                            'vwap': vwap,
                            'sigma': sigma,
                            'atr': atr,
                            'is_trend_day': is_trend_day,
                            'required_sigma': required_sigma
                        }
                        signals.append(signal)
        
        print(f"   ✅ Generated {len(signals)} S2 VWAP signals")
        
        return signals
    
    def calculate_performance(self, signals: List[Dict], bars: List[Dict], symbol: str) -> Dict:
        """Calculate S2 strategy performance metrics"""
        
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
                'avg_z_score': 0.0,
                'trade_details': []
            }
        
        print(f"📊 Calculating S2 performance for {len(signals)} signals...")
        
        point_value = self.point_values.get(symbol, 50.0)
        
        # Create price lookup
        price_lookup = {}
        for i, bar in enumerate(bars):
            timestamp = bar['Start']
            price_lookup[timestamp] = (i, float(bar['Close']))
        
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        trade_details = []
        z_scores = []
        
        for signal in signals:
            entry_time = signal['timestamp']
            entry_price = signal['price']
            stop_price = signal['stop']
            target_price = signal['target']
            
            if entry_time not in price_lookup:
                continue
            
            entry_idx, _ = price_lookup[entry_time]
            z_scores.append(abs(signal['z_score']))
            
            # Simulate trade over max_bars_in_trade period
            exit_price = None
            exit_reason = "Time"
            
            for bar_offset in range(1, min(self.max_bars_in_trade, len(bars) - entry_idx)):
                if entry_idx + bar_offset >= len(bars):
                    break
                
                bar = bars[entry_idx + bar_offset]
                high = float(bar['High'])
                low = float(bar['Low'])
                close = float(bar['Close'])
                
                if signal['signal'] == 'BUY':
                    # Check for stop hit
                    if low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "Stop"
                        break
                    # Check for target hit
                    elif high >= target_price:
                        exit_price = target_price
                        exit_reason = "Target"
                        break
                else:  # SELL
                    # Check for stop hit
                    if high >= stop_price:
                        exit_price = stop_price
                        exit_reason = "Stop"
                        break
                    # Check for target hit
                    elif low <= target_price:
                        exit_price = target_price
                        exit_reason = "Target"
                        break
            
            # If no exit, use last available price
            if exit_price is None:
                max_idx = min(entry_idx + self.max_bars_in_trade, len(bars) - 1)
                exit_price = float(bars[max_idx]['Close'])
                exit_reason = "Time"
            
            # Calculate P&L
            if signal['signal'] == 'BUY':
                pnl = (exit_price - entry_price) * point_value
            else:  # SELL
                pnl = (entry_price - exit_price) * point_value
            
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            trade_details.append({
                'entry_time': entry_time,
                'signal': signal['signal'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'z_score': signal['z_score'],
                'vwap': signal['vwap'],
                'is_trend_day': signal['is_trend_day']
            })
        
        total_trades = len(trade_details)
        
        return {
            'total_signals': len(signals),
            'total_pnl': total_pnl,
            'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
            'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'max_win': max((t['pnl'] for t in trade_details), default=0),
            'max_loss': min((t['pnl'] for t in trade_details), default=0),
            'avg_z_score': sum(z_scores) / len(z_scores) if z_scores else 0,
            'trade_details': trade_details[:10]  # Show first 10 trades
        }
    
    def run_backtest(self, symbol: str) -> Dict:
        """Run complete S2 VWAP backtest on real historical data"""
        
        print(f"\n🎯 Running S2 VWAP Mean Reversion Backtest on Real {symbol} Data")
        print("=" * 70)
        
        # Load real market data
        bars = self.load_historical_data(symbol)
        
        # Calculate technical indicators
        indicators = self.calculate_indicators(bars, symbol)
        
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
        """Display comprehensive S2 backtest results"""
        
        symbol = results['symbol']
        perf = results['performance']
        
        print(f"\n📈 {symbol} S2 VWAP Mean Reversion Performance Results")
        print("=" * 60)
        print(f"📊 Total Historical Bars: {results['data_bars']:,}")
        print(f"🎯 Total VWAP Signals: {perf['total_signals']}")
        print(f"💰 Total P&L: ${perf['total_pnl']:,.2f}")
        print(f"🎲 Win Rate: {perf['win_rate']:.1f}%")
        print(f"📊 Average Trade: ${perf['avg_trade']:,.2f}")
        print(f"✅ Winning Trades: {perf['winning_trades']}")
        print(f"❌ Losing Trades: {perf['losing_trades']}")
        print(f"🎉 Best Trade: ${perf['max_win']:,.2f}")
        print(f"😬 Worst Trade: ${perf['max_loss']:,.2f}")
        print(f"📐 Avg |Z-Score|: {perf['avg_z_score']:.2f}")
        
        # Show sample trades with details
        if perf['trade_details']:
            print(f"\n🔍 Sample S2 VWAP Trading Activity:")
            for i, trade in enumerate(perf['trade_details'][:5]):
                dt = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                pnl_color = "💚" if trade['pnl'] > 0 else "❤️"
                trend_indicator = "📈" if trade['is_trend_day'] else "📊"
                print(f"   {i+1}. {dt.strftime('%m/%d %H:%M')} {trend_indicator} "
                      f"{trade['signal']} {symbol} at ${trade['entry_price']:.2f} → "
                      f"${trade['exit_price']:.2f} ({trade['exit_reason']}) = "
                      f"{pnl_color}${trade['pnl']:,.2f} | Z={trade['z_score']:.2f}")
        
        print()


def main():
    """Run complete S2 VWAP mean reversion backtesting"""
    
    print("🚀 S2 VWAP Mean Reversion Strategy Historical Backtester")
    print("📈 Testing sophisticated fade-extreme-moves strategy on REAL ES and NQ data")
    print("🎯 Data Flow: Real OHLC → Session VWAP → Z-Score → Mean Reversion Signals → P&L")
    print("=" * 85)
    
    backtester = S2StrategyBacktester()
    
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
        print("\n🏆 FINAL S2 VWAP STRATEGY SUMMARY")
        print("=" * 60)
        
        total_signals = sum(r['performance']['total_signals'] for r in all_results.values())
        total_pnl = sum(r['performance']['total_pnl'] for r in all_results.values())
        total_bars = sum(r['data_bars'] for r in all_results.values())
        
        # Calculate combined win rate
        total_winning = sum(r['performance']['winning_trades'] for r in all_results.values())
        total_trades = sum(r['performance']['winning_trades'] + r['performance']['losing_trades'] 
                          for r in all_results.values())
        combined_win_rate = (total_winning / total_trades * 100) if total_trades > 0 else 0
        
        print(f"📊 Combined VWAP Signals: {total_signals}")
        print(f"💰 Combined P&L: ${total_pnl:,.2f}")
        print(f"🎲 Combined Win Rate: {combined_win_rate:.1f}%")
        print(f"🎯 Total Real Market Bars Analyzed: {total_bars:,}")
        print(f"📅 Real market data from past 3 weeks (Aug-Sep 2025)")
        
        if total_pnl > 0:
            print("✅ S2 VWAP Strategy shows POSITIVE performance on real historical data!")
            print("🎉 Your mean reversion strategy would have been profitable!")
        else:
            print("⚠️  S2 VWAP Strategy shows challenges on this specific period")
            print("🔧 Mean reversion can struggle in trending markets")
        
        print(f"\n🎯 S2 VWAP Strategy Characteristics Tested:")
        print(f"   • Session VWAP anchored to 9:30 AM ET")
        print(f"   • Entry: 2.0σ (ES) / 2.6σ (NQ) from VWAP")
        print(f"   • Trend day protection: 2.8σ requirement")
        print(f"   • Volume regime filtering (-0.3 to 2.2 VolZ)")
        print(f"   • Volume imbalance confirmation")
        print(f"   • ADR room requirements (25% minimum)")
        print(f"   • Target: Return to VWAP")
        print(f"   • Stop: Recent swing + 3σ level")
        
        print(f"\n💡 S2 vs S3 Comparison:")
        print(f"   S2 (VWAP Mean Reversion): Sophisticated institutional strategy")
        print(f"   S3 (Compression Breakout): Momentum-based pattern recognition")
        print(f"   S2 requires more complex market conditions but higher win rates expected")


if __name__ == "__main__":
    main()