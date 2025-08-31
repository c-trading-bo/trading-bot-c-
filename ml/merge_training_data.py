#!/usr/bin/env python3
"""
Merge training data from multiple sources for 24/7 cloud ML/RL training.
Combines real trading data, vendor features, and generates dummy data if needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def merge_training_data(data_dir='../data/logs', vendor_dir='../data/vendor'):
    """Merge all available training data sources."""
    
    # Find all available training data sources
    candidate_files = []
    data_dir = Path(data_dir)
    vendor_dir = Path(vendor_dir)

    # Real trading data
    for parquet_file in data_dir.glob('candidates.*.parquet'):
        candidate_files.append(str(parquet_file))
        print(f'Found real data: {parquet_file}')

    # Vendor-generated features
    for parquet_file in vendor_dir.glob('vendor_features_*.parquet'):
        candidate_files.append(str(parquet_file))
        print(f'Found vendor data: {parquet_file}')

    # Generate dummy data if no files found (for testing)
    if not candidate_files:
        print('No training data found, generating dummy data for testing...')
        n = 1000
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': ['ES'] * n,
            'strategy': np.random.choice(['EmaCross', 'MeanReversion', 'Breakout', 'Momentum'], n),
            'session': np.random.choice(['RTH', 'ETH'], n, p=[0.7, 0.3]),
            'regime': np.random.choice(['Range', 'Trend', 'Vol'], n),
            'signal_id': [f'dummy_{i}' for i in range(n)],
            'price': np.random.normal(4500, 100, n),
            'atr': np.random.exponential(20, n),
            'rsi': np.random.uniform(20, 80, n),
            'ema20': np.random.normal(4500, 100, n),
            'ema50': np.random.normal(4500, 100, n),
            'volume': np.random.exponential(1000, n),
            'spread': np.random.exponential(1.0, n),
            'volatility': np.random.exponential(0.02, n),
            'bid_ask_imbalance': np.random.uniform(-0.1, 0.1, n),
            'order_book_imbalance': np.random.uniform(-0.1, 0.1, n),
            'tick_direction': np.random.choice([-1, 0, 1], n),
            'signal_strength': np.random.uniform(0.1, 1.0, n),
            'prior_win_rate': np.random.uniform(0.4, 0.6, n),
            'avg_r_multiple': np.random.normal(0.8, 0.3, n),
            'drawdown_risk': np.random.exponential(0.1, n),
            'news_impact': np.random.exponential(0.05, n),
            'liquidity_risk': np.random.exponential(0.1, n),
            'baseline_multiplier': np.ones(n),
            'label_win': np.random.choice([0, 1], n, p=[0.48, 0.52]),
            'r_multiple': np.random.normal(0.1, 1.5, n),
            'slip_ticks': np.random.exponential(0.5, n)
        })
        
        dummy_file = data_dir / 'candidates.dummy.parquet'
        data_dir.mkdir(parents=True, exist_ok=True)
        dummy_data.to_parquet(dummy_file, index=False)
        candidate_files.append(str(dummy_file))

    # Merge all data sources
    print(f'Merging {len(candidate_files)} data sources...')
    merged_dfs = []
    total_rows = 0
    
    for file in candidate_files:
        try:
            df = pd.read_parquet(file)
            merged_dfs.append(df)
            total_rows += len(df)
            print(f'  - {file}: {len(df)} rows')
        except Exception as e:
            print(f'  - ERROR reading {file}: {e}')

    if not merged_dfs:
        print('ERROR: No valid data files found!')
        sys.exit(1)

    merged_df = pd.concat(merged_dfs, ignore_index=True)
    
    # Deduplicate by signal_id if column exists
    if 'signal_id' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['signal_id'], keep='last')

    print(f'📊 Merged dataset: {len(merged_df)} rows (from {total_rows} total)')
    
    if 'strategy' in merged_df.columns:
        print(f'📊 Strategy distribution: {merged_df["strategy"].value_counts().to_dict()}')
    
    if 'label_win' in merged_df.columns:
        print(f'📊 Win rate: {merged_df["label_win"].mean():.3f}')

    # Save merged dataset
    output_file = data_dir / 'candidates.merged.parquet'
    merged_df.to_parquet(output_file, index=False)
    print(f'✅ Saved merged training data to {output_file}')
    
    return merged_df

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/logs'
    vendor_dir = sys.argv[2] if len(sys.argv) > 2 else '../data/vendor'
    merge_training_data(data_dir, vendor_dir)
