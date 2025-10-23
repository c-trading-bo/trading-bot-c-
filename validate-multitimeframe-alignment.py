#!/usr/bin/env python3
"""
Validate that 1-minute bars align with 5-minute bars.
This script checks data integrity for multi-timeframe trading.

Test criteria:
1. Every 5th 1m bar close time should match a 5m bar close time
2. 5m OHLC should be correctly derived from corresponding 1m bars
3. Check for data gaps in both timeframes
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime (handle timezone)."""
    # Remove timezone for datetime parsing
    ts_without_tz = ts_str.rsplit('-', 1)[0].rsplit('+', 1)[0].strip()
    return datetime.strptime(ts_without_tz, '%Y-%m-%d %H:%M:%S')


def load_bars(file_path):
    """Load bar data from JSON file."""
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data.get('bars', [])


def validate_alignment(symbol):
    """Validate 1m and 5m bar alignment for a symbol."""
    print(f"\n{'='*60}")
    print(f"Validating {symbol} bar alignment...")
    print(f"{'='*60}\n")
    
    data_dir = Path('data/historical')
    
    # Load 5m and 1m bars
    bars_5m = load_bars(data_dir / f'{symbol}_90days.json')
    bars_1m = load_bars(data_dir / f'{symbol}_1m_90days.json')
    
    if not bars_5m:
        print(f"❌ ERROR: No 5m bars found for {symbol}")
        return False
    
    if not bars_1m:
        print(f"❌ ERROR: No 1m bars found for {symbol}")
        return False
    
    print(f"✅ Loaded {len(bars_5m)} 5m bars")
    print(f"✅ Loaded {len(bars_1m)} 1m bars")
    print()
    
    # Create timestamp index for 1m bars
    bars_1m_by_time = {}
    for bar in bars_1m:
        ts = parse_timestamp(bar['timestamp'])
        bars_1m_by_time[ts] = bar
    
    # Validate alignment
    errors = []
    warnings = []
    checked = 0
    aligned = 0
    
    for bar_5m in bars_5m:
        ts_5m = parse_timestamp(bar_5m['timestamp'])
        
        # Find corresponding 1m bars (should be 5 bars ending at ts_5m)
        expected_1m_times = [ts_5m - timedelta(minutes=i) for i in range(4, -1, -1)]
        
        # Check if all 5 1m bars exist
        missing_bars = []
        found_bars = []
        
        for ts_1m in expected_1m_times:
            if ts_1m in bars_1m_by_time:
                found_bars.append(bars_1m_by_time[ts_1m])
            else:
                missing_bars.append(ts_1m)
        
        checked += 1
        
        if len(found_bars) == 5:
            # All 5 1m bars found - validate OHLC
            aligned += 1
            
            # Calculate expected 5m OHLC from 1m bars
            opens = [b['open'] for b in found_bars]
            highs = [b['high'] for b in found_bars]
            lows = [b['low'] for b in found_bars]
            closes = [b['close'] for b in found_bars]
            volumes = [b['volume'] for b in found_bars]
            
            expected_open = opens[0]
            expected_high = max(highs)
            expected_low = min(lows)
            expected_close = closes[-1]
            expected_volume = sum(volumes)
            
            # Compare with 5m bar (allow small floating point differences)
            tolerance = 0.01
            
            if abs(bar_5m['open'] - expected_open) > tolerance:
                errors.append(
                    f"Open mismatch at {ts_5m}: "
                    f"5m={bar_5m['open']}, expected={expected_open}"
                )
            
            if abs(bar_5m['high'] - expected_high) > tolerance:
                errors.append(
                    f"High mismatch at {ts_5m}: "
                    f"5m={bar_5m['high']}, expected={expected_high}"
                )
            
            if abs(bar_5m['low'] - expected_low) > tolerance:
                errors.append(
                    f"Low mismatch at {ts_5m}: "
                    f"5m={bar_5m['low']}, expected={expected_low}"
                )
            
            if abs(bar_5m['close'] - expected_close) > tolerance:
                errors.append(
                    f"Close mismatch at {ts_5m}: "
                    f"5m={bar_5m['close']}, expected={expected_close}"
                )
        
        elif len(found_bars) > 0:
            # Partial match - this is a warning
            warnings.append(
                f"Partial match at {ts_5m}: found {len(found_bars)}/5 1m bars"
            )
        else:
            # No matching 1m bars - likely outside trading hours
            pass
    
    # Report results
    print(f"Alignment Results:")
    print(f"  - Checked: {checked} 5m bars")
    print(f"  - Fully aligned: {aligned} 5m bars")
    print(f"  - Alignment rate: {100*aligned/checked:.1f}%")
    print()
    
    if warnings:
        print(f"⚠️  Warnings: {len(warnings)}")
        for i, warning in enumerate(warnings[:5], 1):
            print(f"  {i}. {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
        print()
    
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for i, error in enumerate(errors[:5], 1):
            print(f"  {i}. {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        print()
        return False
    else:
        print(f"✅ No OHLC mismatches detected!")
        print()
        
        # Show sample aligned bars
        if aligned >= 3:
            print("Sample aligned bars (showing 5m bar and corresponding 1m bars):")
            for i, bar_5m in enumerate(bars_5m[:3], 1):
                ts_5m = parse_timestamp(bar_5m['timestamp'])
                print(f"\n  5m bar {i}: {ts_5m}")
                print(f"    O:{bar_5m['open']} H:{bar_5m['high']} "
                      f"L:{bar_5m['low']} C:{bar_5m['close']} V:{bar_5m['volume']}")
                
                # Show corresponding 1m bars
                expected_1m_times = [ts_5m - timedelta(minutes=i) for i in range(4, -1, -1)]
                for j, ts_1m in enumerate(expected_1m_times, 1):
                    if ts_1m in bars_1m_by_time:
                        bar_1m = bars_1m_by_time[ts_1m]
                        print(f"    1m[{j}] {ts_1m}: "
                              f"O:{bar_1m['open']} H:{bar_1m['high']} "
                              f"L:{bar_1m['low']} C:{bar_1m['close']} V:{bar_1m['volume']}")
        
        return True


def main():
    """Main validation function."""
    print("Multi-Timeframe Bar Alignment Validation")
    print("="*60)
    
    symbols = ['ES', 'NQ']
    all_valid = True
    
    for symbol in symbols:
        valid = validate_alignment(symbol)
        if not valid:
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("✅ VALIDATION PASSED: All bars are properly aligned!")
    else:
        print("❌ VALIDATION FAILED: Some alignment issues detected")
    print("="*60)
    
    return all_valid


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
