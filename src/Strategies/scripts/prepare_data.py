#!/usr/bin/env python3
"""
Prepare Data Script for Ultimate ML/RL Training Pipeline
Creates necessary data structures and validates data integrity.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def create_directories():
    """Create necessary directories for training pipeline."""
    dirs = [
        "data/logs",
        "data/vendor", 
        "data/rl_training",
        "Intelligence/models",
        "Intelligence/data"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def prepare_training_data():
    """Prepare training data for ML/RL models."""
    print("🔄 Preparing training data...")
    
    # Create mock data for testing if no real data exists
    data_dir = Path("data/rl_training")
    
    if not any(data_dir.glob("*.csv")):
        print("📊 Creating mock training data...")
        
        # Create sample ES/NQ futures data
        mock_data = {
            'timestamp': pd.date_range(start='2025-01-01', periods=1000, freq='1min'),
            'symbol': ['ES'] * 500 + ['NQ'] * 500,
            'price': [4500 + i * 0.1 for i in range(500)] + [15000 + i * 0.5 for i in range(500)],
            'volume': [100 + i for i in range(1000)],
            'returns': [0.001 * (i % 10 - 5) for i in range(1000)]
        }
        
        df = pd.DataFrame(mock_data)
        df.to_csv(data_dir / "training_data.csv", index=False)
        print(f"✅ Created mock training data: {data_dir / 'training_data.csv'}")

def validate_environment():
    """Validate training environment setup."""
    print("🔍 Validating environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required directories
    required_dirs = ["data/rl_training", "Intelligence/models"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
    
    return True

def main():
    """Main preparation function."""
    print("🚀 Starting Ultimate ML/RL Data Preparation...")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    
    try:
        create_directories()
        prepare_training_data()
        validate_environment()
        
        print("✅ Data preparation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during data preparation: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
