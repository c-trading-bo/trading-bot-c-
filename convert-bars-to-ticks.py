#!/usr/bin/env python3
"""
Convert 1-minute bars to tick data for backtesting
Since TopStep's lowest granularity is 1-minute bars, this generates realistic
ticks within each bar for simulation purposes.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

def generate_ticks_from_bar(bar, ticks_per_bar=10):
    """
    Generate realistic ticks from a 1-minute bar
    
    Args:
        bar: Dict with keys: timestamp, open, high, low, close, volume
        ticks_per_bar: Number of ticks to generate per bar (default 10)
    
    Returns:
        List of tick Quote objects
    """
    ticks = []
    
    # Parse timestamp
    bar_time = datetime.fromisoformat(bar['timestamp'].replace('Z', '+00:00'))
    
    # Bar data
    open_price = float(bar['open'])
    high_price = float(bar['high'])
    low_price = float(bar['low'])
    close_price = float(bar['close'])
    total_volume = int(bar['volume'])
    
    # Generate price path through the bar
    # Start at open, touch high and low, end at close
    prices = []
    
    # Determine if we go high first or low first
    if abs(high_price - open_price) > abs(open_price - low_price):
        # Touch high first
        prices.append(open_price)
        prices.append(high_price)
        prices.append(low_price)
        prices.append(close_price)
    else:
        # Touch low first  
        prices.append(open_price)
        prices.append(low_price)
        prices.append(high_price)
        prices.append(close_price)
    
    # Interpolate to get ticks_per_bar total prices
    if ticks_per_bar > 4:
        # Add interpolated prices between key levels
        interpolated = []
        for i in range(len(prices) - 1):
            steps = ticks_per_bar // 3
            for j in range(steps):
                ratio = j / steps
                price = prices[i] + (prices[i+1] - prices[i]) * ratio
                interpolated.append(price)
        prices = interpolated[:ticks_per_bar]
    
    # Ensure we have exactly ticks_per_bar prices
    while len(prices) < ticks_per_bar:
        prices.append(close_price)
    prices = prices[:ticks_per_bar]
    
    # Distribute volume across ticks
    volumes = []
    remaining_volume = total_volume
    for i in range(ticks_per_bar - 1):
        vol = random.randint(1, max(2, remaining_volume // (ticks_per_bar - i)))
        volumes.append(vol)
        remaining_volume -= vol
    volumes.append(max(1, remaining_volume))  # Last tick gets remaining
    
    # Generate ticks
    time_delta_seconds = 60 / ticks_per_bar  # Spread across the minute
    
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        tick_time = bar_time + timedelta(seconds=i * time_delta_seconds)
        
        # Generate bid/ask spread (0.25 point spread typical for ES)
        spread = 0.25
        bid = round(price - spread/2, 2)
        ask = round(price + spread/2, 2)
        
        tick = {
            "Time": tick_time.isoformat().replace('+00:00', 'Z'),
            "Symbol": "ES",  # Will be set properly by caller
            "Bid": bid,
            "Ask": ask,
            "Last": round(price, 2),
            "Volume": volume,
            "Open": round(price, 2),
            "High": round(price, 2),
            "Low": round(price, 2),
            "Close": round(price, 2)
        }
        ticks.append(tick)
    
    return ticks

def convert_bars_to_quotes(bars_file, output_file, symbol="ES", ticks_per_bar=10):
    """
    Convert 1-minute bars from TopStep format to Quote ticks for backtesting
    
    Args:
        bars_file: Input file with bars_1m data
        output_file: Output file for quotes
        symbol: Trading symbol
        ticks_per_bar: Number of ticks to generate per bar
    """
    print(f"Converting 1-minute bars to ticks for {symbol}...")
    print(f"Input: {bars_file}")
    print(f"Output: {output_file}")
    print(f"Ticks per bar: {ticks_per_bar}")
    print()
    
    # Load source data
    with open(bars_file, 'r') as f:
        data = json.load(f)
    
    bars_1m = data.get('bars_1m', [])
    
    if not bars_1m:
        print(f"Warning: No bars_1m data found in {bars_file}")
        return
    
    print(f"Found {len(bars_1m)} 1-minute bars")
    
    # Generate ticks from each bar
    all_quotes = []
    for bar in bars_1m:
        ticks = generate_ticks_from_bar(bar, ticks_per_bar)
        for tick in ticks:
            tick['Symbol'] = symbol
        all_quotes.extend(ticks)
    
    print(f"Generated {len(all_quotes)} ticks from {len(bars_1m)} bars")
    print(f"Average: {len(all_quotes)/len(bars_1m):.1f} ticks per bar")
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(all_quotes, f, indent=2)
    
    print(f"✅ Saved to {output_file}")
    print()
    
    # Show sample
    if all_quotes:
        print("Sample ticks from first bar:")
        for i, tick in enumerate(all_quotes[:3]):
            print(f"  Tick {i+1}: {tick['Time']} | Price: ${tick['Last']:.2f} | Bid: ${tick['Bid']:.2f} | Ask: ${tick['Ask']:.2f} | Vol: {tick['Volume']}")

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent / "datasets"
    quotes_dir = datasets_dir / "quotes"
    quotes_dir.mkdir(exist_ok=True)
    
    # Convert ES 1-minute bars to ticks
    es_source = datasets_dir / "ES_90days.json"
    if es_source.exists():
        convert_bars_to_quotes(
            es_source, 
            quotes_dir / "es_quotes.json",
            symbol="ES",
            ticks_per_bar=10  # 10 ticks per 1-minute bar
        )
    
    # Convert NQ 1-minute bars to ticks
    nq_source = datasets_dir / "NQ_90days.json"
    if nq_source.exists():
        convert_bars_to_quotes(
            nq_source,
            quotes_dir / "nq_quotes.json", 
            symbol="NQ",
            ticks_per_bar=10
        )
    
    print("\n✅ Conversion complete!")
    print("Backtesting will now use ticks generated from 1-minute bars")
    print("This matches TopStep's actual data granularity (1-minute bars)")
