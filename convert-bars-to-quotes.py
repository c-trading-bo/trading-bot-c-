#!/usr/bin/env python3
"""
Convert 1-minute bars directly to Quote format (NO SIMULATION)
Each bar becomes a single Quote data point - no interpolation or simulation.

This is the simple approach until real tick data is available from TopStep.
"""
import json
from pathlib import Path
from datetime import datetime

def convert_bar_to_quote(bar, symbol="ES"):
    """
    Convert a single 1-minute bar to a Quote object
    No simulation - just use the bar's OHLC data as-is
    
    Args:
        bar: Dict with keys: timestamp, open, high, low, close, volume
        symbol: Trading symbol
    
    Returns:
        Quote dict
    """
    # Use close price as the "Last" price for this bar
    last_price = float(bar['close'])
    
    # Generate bid/ask from close (simple 0.25 spread for ES)
    spread = 0.25
    bid = round(last_price - spread/2, 2)
    ask = round(last_price + spread/2, 2)
    
    quote = {
        "Time": bar['timestamp'],
        "Symbol": symbol,
        "Bid": bid,
        "Ask": ask,
        "Last": round(last_price, 2),
        "Volume": int(bar['volume']),
        "Open": round(float(bar['open']), 2),
        "High": round(float(bar['high']), 2),
        "Low": round(float(bar['low']), 2),
        "Close": round(float(bar['close']), 2)
    }
    
    return quote

def convert_bars_to_quotes_simple(bars_file, output_file, symbol="ES"):
    """
    Convert 1-minute bars to quotes WITHOUT simulation
    Each bar = 1 quote (no interpolation)
    
    Args:
        bars_file: Input file with bars_1m data
        output_file: Output file for quotes
        symbol: Trading symbol
    """
    print(f"Converting 1-minute bars to quotes for {symbol}...")
    print(f"Mode: DIRECT (no simulation - each bar = 1 quote)")
    print(f"Input: {bars_file}")
    print(f"Output: {output_file}")
    print()
    
    # Load source data
    with open(bars_file, 'r') as f:
        data = json.load(f)
    
    bars_1m = data.get('bars_1m', [])
    
    if not bars_1m:
        print(f"Warning: No bars_1m data found in {bars_file}")
        return
    
    print(f"Found {len(bars_1m)} 1-minute bars")
    
    # Convert each bar directly to a quote (NO simulation)
    quotes = []
    for bar in bars_1m:
        quote = convert_bar_to_quote(bar, symbol)
        quotes.append(quote)
    
    print(f"Generated {len(quotes)} quotes from {len(bars_1m)} bars")
    print(f"Ratio: 1 quote per bar (NO simulation/interpolation)")
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(quotes, f, indent=2)
    
    print(f"✅ Saved to {output_file}")
    print()
    
    # Show sample
    if quotes:
        print("Sample quotes (first 3 bars):")
        for i, quote in enumerate(quotes[:3]):
            print(f"  Bar {i+1}: {quote['Time']} | Close: ${quote['Last']:.2f} | OHLC: ${quote['Open']:.2f}/${quote['High']:.2f}/${quote['Low']:.2f}/${quote['Close']:.2f} | Vol: {quote['Volume']}")

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent / "datasets"
    quotes_dir = datasets_dir / "quotes"
    quotes_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("1-MINUTE BAR TO QUOTE CONVERSION (NO SIMULATION)")
    print("=" * 70)
    print()
    print("Each 1-minute bar becomes ONE quote data point.")
    print("No simulation, no interpolation - just direct conversion.")
    print("Waiting for real tick data from TopStep.")
    print()
    
    # Convert ES 1-minute bars to quotes
    es_source = datasets_dir / "ES_90days.json"
    if es_source.exists():
        convert_bars_to_quotes_simple(
            es_source, 
            quotes_dir / "es_quotes.json",
            symbol="ES"
        )
    
    # Convert NQ 1-minute bars to quotes
    nq_source = datasets_dir / "NQ_90days.json"
    if nq_source.exists():
        convert_bars_to_quotes_simple(
            nq_source,
            quotes_dir / "nq_quotes.json", 
            symbol="NQ"
        )
    
    print()
    print("✅ Conversion complete!")
    print("Each bar = 1 quote (no simulation)")
    print("Ready for backtesting with 1-minute bar granularity")
