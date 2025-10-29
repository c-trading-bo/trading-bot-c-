#!/usr/bin/env python3
"""
Generate realistic 1-minute bar data (simulating TopStep data)
Creates sample data that represents what TopStep API would provide

TopStep Note: TopStep's API provides 1-minute bars as the lowest granularity.
This script generates realistic 1-minute OHLCV bars that match TopStep's data format.
"""
import json
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_1min_bars(symbol, start_time, num_bars=60, base_price=4700.0):
    """
    Generate realistic 1-minute bars
    Simulates what TopStep API provides (1-min is their lowest granularity)
    """
    bars = []
    current_time = start_time
    current_price = base_price
    
    for i in range(num_bars):
        # Random walk with mean reversion
        price_change = random.gauss(0, 0.5)  # Small changes per minute
        if i > 0:
            mean_reversion = (base_price - current_price) * 0.05
            price_change += mean_reversion
        
        # OHLC for the minute
        open_price = current_price
        
        # High and low around open
        high_move = abs(random.gauss(0, 0.75))
        low_move = abs(random.gauss(0, 0.75))
        
        high_price = open_price + high_move
        low_price = open_price - low_move
        
        # Close somewhere between high and low
        close_price = open_price + price_change
        close_price = max(low_price, min(high_price, close_price))
        
        # Ensure high is actually highest and low is lowest
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume (higher during volatile moves)
        volatility = (high_price - low_price) / open_price
        base_volume = 150
        volume = int(base_volume * (1 + volatility * 10) * random.uniform(0.8, 1.2))
        
        bar = {
            "timestamp": current_time.isoformat() + "Z",
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        }
        
        bars.append(bar)
        
        # Move to next minute
        current_time += timedelta(minutes=1)
        current_price = close_price
    
    return bars

def create_topstep_format_data(symbol, num_bars=60):
    """Create data in TopStep-like format with bars_1m"""
    # Start time (market open)
    start_time = datetime(2024, 10, 29, 9, 30, 0)
    
    base_price = 4700.0 if symbol == "ES" else 15000.0
    
    # Generate 1-minute bars (TopStep's lowest granularity)
    bars_1m = generate_1min_bars(symbol, start_time, num_bars, base_price)
    
    # Also generate 5-minute bars by aggregating
    bars_5m = []
    for i in range(0, len(bars_1m), 5):
        chunk = bars_1m[i:i+5]
        if chunk:
            bar_5m = {
                "timestamp": chunk[0]['timestamp'],
                "open": chunk[0]['open'],
                "high": max(b['high'] for b in chunk),
                "low": min(b['low'] for b in chunk),
                "close": chunk[-1]['close'],
                "volume": sum(b['volume'] for b in chunk)
            }
            bars_5m.append(bar_5m)
    
    data = {
        "symbol": symbol,
        "bars_1m": bars_1m,
        "bars_5m": bars_5m,
        "metadata": {
            "source": "Simulated TopStep format",
            "note": "1-minute bars are TopStep's lowest granularity",
            "bars_1m_count": len(bars_1m),
            "bars_5m_count": len(bars_5m)
        }
    }
    
    return data

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    print("Generating realistic 1-minute bar data (TopStep format)...")
    print()
    
    # Generate ES data (60 bars = 1 hour of data)
    print("📊 Generating ES 1-minute bars...")
    es_data = create_topstep_format_data("ES", num_bars=60)
    es_file = datasets_dir / "ES_90days.json"
    with open(es_file, 'w') as f:
        json.dump(es_data, f, indent=2)
    print(f"   ✅ Created {es_file}")
    print(f"   📈 {len(es_data['bars_1m'])} 1-minute bars")
    print(f"   📈 {len(es_data['bars_5m'])} 5-minute bars")
    print(f"   💰 Price range: ${es_data['bars_1m'][0]['open']:.2f} - ${es_data['bars_1m'][-1]['close']:.2f}")
    print()
    
    # Generate NQ data
    print("📊 Generating NQ 1-minute bars...")
    nq_data = create_topstep_format_data("NQ", num_bars=60)
    nq_file = datasets_dir / "NQ_90days.json"
    with open(nq_file, 'w') as f:
        json.dump(nq_data, f, indent=2)
    print(f"   ✅ Created {nq_file}")
    print(f"   📈 {len(nq_data['bars_1m'])} 1-minute bars")
    print(f"   📈 {len(nq_data['bars_5m'])} 5-minute bars")
    print(f"   💰 Price range: ${nq_data['bars_1m'][0]['open']:.2f} - ${nq_data['bars_1m'][-1]['close']:.2f}")
    print()
    
    print("✅ Sample data generation complete!")
    print()
    print("Next step: Run convert-bars-to-ticks.py to generate tick data from these 1-minute bars")
