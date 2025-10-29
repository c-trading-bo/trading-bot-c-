#!/usr/bin/env python3
"""
Convert tick data from ES_90days.json format to Quote format for backtest provider
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def convert_tick_data(input_file, output_file):
    """Convert tick data to Quote array format"""
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    symbol = data.get('symbol', 'ES')
    ticks = data.get('ticks', [])
    
    if not ticks:
        print(f"Warning: No ticks found in {input_file}")
        return
    
    quotes = []
    for tick in ticks:
        quote = {
            "Time": tick['timestamp'],
            "Symbol": symbol,
            "Bid": float(tick.get('bid', tick['price'] - 0.25)),
            "Ask": float(tick.get('ask', tick['price'] + 0.25)),
            "Last": float(tick['price']),
            "Volume": int(tick.get('size', 1)),
            "Open": float(tick['price']),
            "High": float(tick['price']),
            "Low": float(tick['price']),
            "Close": float(tick['price'])
        }
        quotes.append(quote)
    
    print(f"Converted {len(quotes)} ticks")
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(quotes, f, indent=2)
    
    print(f"Done! Created {output_file} with {len(quotes)} quotes")

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent / "datasets"
    quotes_dir = datasets_dir / "quotes"
    quotes_dir.mkdir(exist_ok=True)
    
    # Convert ES data
    es_input = datasets_dir / "ES_90days.json"
    if es_input.exists():
        convert_tick_data(es_input, quotes_dir / "es_quotes.json")
    else:
        print(f"Warning: {es_input} not found")
    
    # Convert NQ data
    nq_input = datasets_dir / "NQ_90days.json"
    if nq_input.exists():
        convert_tick_data(nq_input, quotes_dir / "nq_quotes.json")
    else:
        print(f"Warning: {nq_input} not found")
