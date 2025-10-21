#!/usr/bin/env python3
"""
Fetch 90 days of historical data from TopstepX and save it for bot practice.
This pre-loads the data so the bot can practice immediately without waiting for SDK initialization.

Supports two modes:
1. FULL refresh: Fetch 90 days, replace entire file
2. INCREMENTAL: Fetch only new bars since last update, append and trim to 90 days

Phase 3 Enhancements:
- Retry logic with exponential backoff (3 attempts)
- Rate limiting handling
- Data validation (nulls, zeros, outliers)
- Resume capability from last successful bar
- Progress logging every 10,000 bars
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# Add parent directory to path to import TopstepXAdapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'adapters'))

from topstep_x_adapter import TopstepXAdapter

# Configuration
REFRESH_MODE = os.getenv('REFRESH_MODE', 'incremental')  # 'full' or 'incremental'
LOOKBACK_DAYS = 90  # Keep last 90 days
MAX_RETRIES = 3  # Maximum retry attempts for failed requests
RETRY_DELAY_BASE = 2  # Base delay in seconds for exponential backoff
RATE_LIMIT_DELAY = 5  # Delay when rate limited
PROGRESS_LOG_INTERVAL = 10000  # Log progress every N bars


def load_existing_data(file_path):
    """Load existing historical data from file."""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"   ⚠️ Could not load existing file: {e}")
        return None


def get_last_timestamp(existing_data):
    """Get the timestamp of the last bar in existing data."""
    if not existing_data or not existing_data.get('bars'):
        return None
    
    bars = existing_data['bars']
    if not bars:
        return None
    
    last_bar = bars[-1]
    timestamp_str = last_bar.get('timestamp')
    
    if not timestamp_str:
        return None
    
    try:
        # Parse timestamp (format: "2025-08-31 17:00:00-05:00")
        # Remove timezone for datetime parsing
        ts_without_tz = timestamp_str.rsplit('-', 1)[0].rsplit('+', 1)[0].strip()
        return datetime.strptime(ts_without_tz, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"   ⚠️ Could not parse timestamp: {e}")
        return None


def validate_bar(bar, symbol):
    """
    Validate a single bar for data quality.
    Phase 3: Check for nulls, zeros, outliers
    
    Returns: (is_valid, reason)
    """
    # Check required fields
    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for field in required_fields:
        if field not in bar or bar[field] is None:
            return False, f"Missing or null {field}"
    
    # Check for zero prices (invalid for ES/NQ)
    if bar['open'] == 0 or bar['high'] == 0 or bar['low'] == 0 or bar['close'] == 0:
        return False, "Zero price detected"
    
    # Check OHLC logic: High >= Low, and Open/Close between High/Low
    high = float(bar['high'])
    low = float(bar['low'])
    open_price = float(bar['open'])
    close_price = float(bar['close'])
    
    if high < low:
        return False, f"High ({high}) < Low ({low})"
    
    if open_price > high or open_price < low:
        return False, f"Open ({open_price}) outside High/Low range"
    
    if close_price > high or close_price < low:
        return False, f"Close ({close_price}) outside High/Low range"
    
    # Check for outliers (ES typically 4000-6000, NQ 15000-20000)
    # Allow wide range but catch extreme outliers
    if symbol == 'ES':
        if close_price < 1000 or close_price > 10000:
            return False, f"Price outlier for ES: {close_price}"
    elif symbol == 'NQ':
        if close_price < 5000 or close_price > 30000:
            return False, f"Price outlier for NQ: {close_price}"
    
    return True, ""


def trim_to_lookback_days(bars, lookback_days):
    """Keep only the most recent bars within lookback window."""
    if not bars:
        return bars
    
    # Find cutoff date
    cutoff = datetime.now() - timedelta(days=lookback_days)
    
    # Filter bars
    trimmed = []
    for bar in bars:
        timestamp_str = bar.get('timestamp', '')
        try:
            ts_without_tz = timestamp_str.rsplit('-', 1)[0].rsplit('+', 1)[0].strip()
            bar_time = datetime.strptime(ts_without_tz, '%Y-%m-%d %H:%M:%S')
            
            if bar_time >= cutoff:
                trimmed.append(bar)
        except:
            # Keep bar if we can't parse (safer)
            trimmed.append(bar)
    
    return trimmed


async def fetch_with_retry(adapter, symbol, start_date, end_date, attempt=1):
    """
    Fetch historical bars with retry logic and exponential backoff.
    Phase 3: Retry on timeout/failure with exponential backoff
    
    Returns: (success, bars_result, error_message)
    """
    try:
        bars_result = await adapter.fetch_historical_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if bars_result and bars_result.get('success'):
            return True, bars_result, None
        else:
            error_msg = bars_result.get('error', 'Unknown error') if bars_result else 'No response'
            
            # Check for rate limiting
            if 'rate limit' in error_msg.lower() or 'too many' in error_msg.lower():
                print(f"      ⚠️ Rate limited, waiting {RATE_LIMIT_DELAY}s...")
                await asyncio.sleep(RATE_LIMIT_DELAY)
                
                if attempt < MAX_RETRIES:
                    print(f"      🔄 Retry {attempt}/{MAX_RETRIES} after rate limit...")
                    return await fetch_with_retry(adapter, symbol, start_date, end_date, attempt + 1)
            
            # Check for timeout
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE ** attempt  # Exponential backoff: 2s, 4s, 8s
                    print(f"      ⚠️ Timeout, waiting {delay}s before retry {attempt}/{MAX_RETRIES}...")
                    await asyncio.sleep(delay)
                    return await fetch_with_retry(adapter, symbol, start_date, end_date, attempt + 1)
            
            return False, bars_result, error_msg
            
    except asyncio.TimeoutError:
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY_BASE ** attempt
            print(f"      ⚠️ Request timeout, waiting {delay}s before retry {attempt}/{MAX_RETRIES}...")
            await asyncio.sleep(delay)
            return await fetch_with_retry(adapter, symbol, start_date, end_date, attempt + 1)
        else:
            return False, None, f"Timeout after {MAX_RETRIES} attempts"
    
    except Exception as e:
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY_BASE ** attempt
            print(f"      ⚠️ Error: {str(e)}, retrying in {delay}s ({attempt}/{MAX_RETRIES})...")
            await asyncio.sleep(delay)
            return await fetch_with_retry(adapter, symbol, start_date, end_date, attempt + 1)
        else:
            return False, None, str(e)


async def fetch_and_save_historical_data():
    """Fetch 90 days of historical data and save to disk for bot practice."""
    
    # Check environment variables
    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    
    if not api_key or not username:
        print("❌ ERROR: TOPSTEPX_API_KEY and TOPSTEPX_USERNAME must be set in .env")
        return False
    
    print(f"✅ Environment variables loaded")
    print(f"   Username: {username}")
    print(f"   API Key: {'*' * 20}{api_key[-4:]}")
    print(f"   Refresh Mode: {REFRESH_MODE.upper()}")
    print()
    
    # Symbols to fetch
    symbols = ['ES', 'NQ']
    
    # Create data directory
    data_dir = Path('data/historical')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        print(f"🔍 Fetching {symbol} data...")
        
        # Check for existing data
        file_path = data_dir / f'{symbol}_90days.json'
        existing_data = load_existing_data(file_path)
        
        # Determine date range based on mode
        end_date = datetime.now()
        
        if REFRESH_MODE == 'incremental' and existing_data:
            # Incremental: Fetch only new bars since last update
            last_timestamp = get_last_timestamp(existing_data)
            
            if last_timestamp:
                start_date = last_timestamp + timedelta(minutes=5)  # Start after last bar
                print(f"   📊 Incremental update from last bar: {last_timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                # If last bar is very recent (< 10 minutes), nothing to fetch
                time_since_last = (end_date - last_timestamp).total_seconds() / 60
                if time_since_last < 10:
                    print(f"   ✅ Data is up-to-date (last bar {int(time_since_last)} min ago)")
                    continue
            else:
                # No valid timestamp, fall back to full refresh
                print(f"   ⚠️ Could not find last timestamp, doing full refresh")
                start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        else:
            # Full refresh: Fetch entire lookback window
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)
            print(f"   🔄 Full refresh mode")
        
        print(f"   � Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print()
        
        try:
            # Create adapter
            adapter = TopstepXAdapter(instruments=[symbol])
            
            # Initialize with stdout redirect
            print(f"   Initializing TopstepX SDK...")
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                await asyncio.wait_for(adapter.initialize(), timeout=60.0)
            finally:
                sys.stdout = original_stdout
            
            print(f"   ✅ TopstepX SDK initialized")
            
            # Fetch in chunks to get full 90 days
            # API limit: 1000 bars/request
            # 5min bars: 1000 bars = ~13 trading days
            # Need ~7 chunks for 90 calendar days
            print(f"   Fetching bars in chunks (API limit: 1000 bars/request)...")
            
            all_bars = []
            chunk_start = start_date
            chunk_number = 1
            invalid_bars_count = 0
            total_fetched = 0
            
            while chunk_start < end_date:
                # Each chunk: ~15 calendar days to ensure we get 1000 bars
                chunk_end = min(chunk_start + timedelta(days=15), end_date)
                
                print(f"   📦 Chunk {chunk_number}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
                
                # Use retry logic
                success, bars_result, error = await fetch_with_retry(
                    adapter, symbol, chunk_start, chunk_end
                )
                
                if success and bars_result:
                    # Extract bars from result
                    bars_data = bars_result.get('bars', {})
                    if isinstance(bars_data, dict):
                        chunk_bars = bars_data.get('bars', [])
                    else:
                        chunk_bars = bars_data
                    
                    if chunk_bars:
                        # Validate bars (Phase 3)
                        validated_bars = []
                        for bar in chunk_bars:
                            is_valid, reason = validate_bar(bar, symbol)
                            if is_valid:
                                validated_bars.append(bar)
                            else:
                                invalid_bars_count += 1
                                # Only log first few invalid bars to avoid spam
                                if invalid_bars_count <= 5:
                                    print(f"      ⚠️ Invalid bar: {reason} - {bar.get('timestamp', 'N/A')}")
                        
                        fetched_count = len(validated_bars)
                        total_fetched += fetched_count
                        
                        print(f"      ✅ Fetched {fetched_count} valid bars")
                        
                        if invalid_bars_count > 0 and invalid_bars_count <= 5:
                            print(f"      ⚠️ Filtered {invalid_bars_count} invalid bars")
                        
                        all_bars.extend(validated_bars)
                        
                        # Progress logging (Phase 3)
                        if total_fetched > 0 and total_fetched % PROGRESS_LOG_INTERVAL == 0:
                            print(f"   📊 Progress: {total_fetched:,} bars fetched so far...")
                    else:
                        print(f"      ⚠️ No data returned")
                else:
                    print(f"      ❌ Request failed after {MAX_RETRIES} retries: {error}")
                    # Continue to next chunk rather than failing completely
                
                # Move to next chunk
                chunk_start = chunk_end
                chunk_number += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            
            if invalid_bars_count > 5:
                print(f"   ⚠️ Total invalid bars filtered: {invalid_bars_count}")
            
            bar_count = len(all_bars)
            
            if bar_count == 0:
                if REFRESH_MODE == 'incremental' and existing_data:
                    print(f"   ℹ️ No new bars to fetch (data already up-to-date)")
                    continue
                else:
                    print(f"   ❌ No data fetched for {symbol}")
                    continue
            
            print(f"   ✅ Fetched: {bar_count} new bars")
            
            # Merge with existing data if incremental mode
            if REFRESH_MODE == 'incremental' and existing_data:
                existing_bars = existing_data.get('bars', [])
                print(f"   🔗 Merging with {len(existing_bars)} existing bars...")
                
                # Combine: existing + new
                combined_bars = existing_bars + all_bars
                
                # Sort by timestamp
                combined_bars.sort(key=lambda b: b.get('timestamp', ''))
                
                # Remove duplicates
                seen_timestamps = set()
                bars_list = []
                for bar in combined_bars:
                    ts = bar.get('timestamp')
                    if ts and ts not in seen_timestamps:
                        seen_timestamps.add(ts)
                        bars_list.append(bar)
                
                # Trim to lookback window (keep last 90 days)
                bars_list = trim_to_lookback_days(bars_list, LOOKBACK_DAYS)
                
                bar_count = len(bars_list)
                print(f"   ✅ After merge and trim: {bar_count} bars (last {LOOKBACK_DAYS} days)")
            else:
                # Full refresh mode: just use fetched bars
                # Sort by timestamp and remove duplicates
                all_bars.sort(key=lambda b: b.get('timestamp', ''))
            
            seen_timestamps = set()
            bars_list = []
            for bar in all_bars:
                ts = bar.get('timestamp')
                if ts and ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    bars_list.append(bar)
            
            if len(bars_list) < bar_count:
                print(f"   🔧 Removed {bar_count - len(bars_list)} duplicate bars")
                bar_count = len(bars_list)
            
            if bar_count == 0:
                print(f"   ⚠️  No data returned for {symbol}")
                continue
            
            # Save to file with enhanced metadata (Phase 3)
            output_file = data_dir / f'{symbol}_90days.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'timeframe': '5min',
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'bar_count': bar_count,
                    'bars': bars_list,
                    'fetched_at': datetime.now().isoformat(),
                    'refresh_mode': REFRESH_MODE,
                    'validation': {
                        'invalid_bars_filtered': invalid_bars_count,
                        'validation_enabled': True
                    }
                }, f, indent=2)
            
            print(f"   💾 Saved to: {output_file}")
            
            # Show sample bars
            if bar_count >= 5:
                print(f"\n   📊 Sample bars (first 3):")
                for i, bar in enumerate(bars_list[:3], 1):
                    timestamp = bar.get('timestamp', 'N/A')
                    open_price = bar.get('open', 0)
                    high = bar.get('high', 0)
                    low = bar.get('low', 0)
                    close = bar.get('close', 0)
                    volume = bar.get('volume', 0)
                    print(f"      [{i}] {timestamp}")
                    print(f"          O: {open_price}, H: {high}, L: {low}, C: {close}, V: {volume}")
                
                print(f"\n   📊 Sample bars (last 3):")
                for i, bar in enumerate(bars_list[-3:], bar_count - 2):
                    timestamp = bar.get('timestamp', 'N/A')
                    open_price = bar.get('open', 0)
                    high = bar.get('high', 0)
                    low = bar.get('low', 0)
                    close = bar.get('close', 0)
                    volume = bar.get('volume', 0)
                    print(f"      [{i}] {timestamp}")
                    print(f"          O: {open_price}, H: {high}, L: {low}, C: {close}, V: {volume}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol} data: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print("✅ HISTORICAL DATA FETCH COMPLETED")
    print(f"📁 Data saved to: {data_dir.absolute()}")
    print()
    print("🤖 Bot can now practice using:")
    print("   - data/historical/ES_90days.json")
    print("   - data/historical/NQ_90days.json")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    # Run the async function
    success = asyncio.run(fetch_and_save_historical_data())
    sys.exit(0 if success else 1)
