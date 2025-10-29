#!/usr/bin/env python3
"""
Generate sample tick data for backtesting demonstration
Creates realistic tick-by-tick data that simulates market movement
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_realistic_ticks(symbol, base_price, num_ticks=500, start_time=None):
    """Generate realistic tick data with bid/ask spread and volume"""
    if start_time is None:
        start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    
    ticks = []
    current_price = base_price
    current_time = start_time
    
    # Market microstructure parameters
    spread = base_price * 0.0001  # 0.01% spread (realistic for ES futures)
    volatility = base_price * 0.0002  # Price volatility per tick
    
    for i in range(num_ticks):
        # Random walk with mean reversion
        price_change = random.gauss(0, volatility)
        if i > 0:
            # Add mean reversion (prices tend to revert to starting price)
            mean_reversion = (base_price - current_price) * 0.01
            price_change += mean_reversion
        
        current_price += price_change
        
        # Generate bid/ask with realistic spread
        mid_price = round(current_price, 2)
        half_spread = spread / 2
        bid = round(mid_price - half_spread, 2)
        ask = round(mid_price + half_spread, 2)
        last = mid_price  # Last trade at mid
        
        # Volume varies randomly (more volume during volatile moves)
        volume = max(1, int(random.expovariate(1/10)))  # Exponential distribution
        
        # Time advances by 1-5 seconds per tick (realistic for futures)
        time_delta = timedelta(seconds=random.randint(1, 5))
        current_time += time_delta
        
        quote = {
            "Time": current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "Symbol": symbol,
            "Bid": bid,
            "Ask": ask,
            "Last": last,
            "Volume": volume,
            "Open": last,
            "High": last,
            "Low": last,
            "Close": last
        }
        ticks.append(quote)
    
    return ticks

def main():
    """Generate sample tick data for ES and NQ"""
    datasets_dir = Path(__file__).parent / "datasets"
    quotes_dir = datasets_dir / "quotes"
    quotes_dir.mkdir(exist_ok=True)
    
    print("Generating realistic tick data for backtesting...")
    
    # Generate ES ticks (E-mini S&P 500)
    print("\n📊 Generating ES (E-mini S&P 500) ticks...")
    es_ticks = generate_realistic_ticks("ES", base_price=4700.0, num_ticks=500)
    es_output = quotes_dir / "es_quotes.json"
    with open(es_output, 'w') as f:
        json.dump(es_ticks, f, indent=2)
    print(f"   ✅ Created {es_output} with {len(es_ticks)} ticks")
    print(f"   📈 Price range: ${es_ticks[0]['Last']:.2f} - ${es_ticks[-1]['Last']:.2f}")
    
    # Generate NQ ticks (E-mini NASDAQ)
    print("\n📊 Generating NQ (E-mini NASDAQ) ticks...")
    nq_ticks = generate_realistic_ticks("NQ", base_price=15000.0, num_ticks=500)
    nq_output = quotes_dir / "nq_quotes.json"
    with open(nq_output, 'w') as f:
        json.dump(nq_ticks, f, indent=2)
    print(f"   ✅ Created {nq_output} with {len(nq_ticks)} ticks")
    print(f"   📈 Price range: ${nq_ticks[0]['Last']:.2f} - ${nq_ticks[-1]['Last']:.2f}")
    
    print("\n✨ Sample data generation complete!")
    print(f"   You can now run backtest mode without API access")

if __name__ == "__main__":
    main()
