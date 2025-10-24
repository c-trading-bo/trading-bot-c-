#!/usr/bin/env python3
"""
Convert unified_brain_training_data.json to TradingExperience format
for the ExperienceRepository to load during Lab Mode training.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import uuid

def convert_to_trading_experience(entry, index):
    """Convert a unified brain training data entry to TradingExperience format"""
    
    # Parse timestamp
    timestamp = entry.get('timestamp', datetime.utcnow().isoformat() + 'Z')
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    # Extract features
    features = entry.get('features', {})
    
    # Map to TradingExperience structure
    experience = {
        "experienceId": str(uuid.uuid4()),
        "timestamp": timestamp,
        "positionId": f"historical_{index}",
        
        # STATE at entry
        "entryRegime": "TREND" if features.get('regime', 0) == 1 else "RANGE",
        "entryRegimeConfidence": 0.75,
        "entryConfidence": 0.8,
        "symbol": "ES",  # Assume ES for historical data
        "entryHour": features.get('hour', 0),
        "entryDayOfWeek": features.get('day_of_week', 0),
        "volatilityAtEntry": features.get('volatility', 0.0001),
        
        # ACTION
        "strategy": entry.get('strategy', 'S2'),
        "positionSize": 1 if entry.get('action', 'HOLD') == 'BUY' else (-1 if entry.get('action') == 'SELL' else 0),
        "entryPrice": features.get('price', 0),
        "initialStopPrice": features.get('price', 0) - (features.get('atr', 2.0) * 2),
        "initialTargetPrice": features.get('price', 0) + (features.get('atr', 2.0) * 3),
        "breakevenAfterTicks": 8,
        "trailTicks": 6,
        
        # REWARD
        "rMultiple": entry.get('reward', 0) * 2.0,  # Convert reward to R-multiple
        "pnL": entry.get('pnl', 0),
        "sharpeContribution": entry.get('reward', 0) / max(features.get('volatility', 0.0001), 0.0001),
        "exitReason": "Target" if entry.get('reward', 0) > 0 else ("StopLoss" if entry.get('reward', 0) < 0 else "Time"),
        "durationMinutes": 15.0,  # Assume 15 min avg trade
        
        # NEXT STATE at exit
        "exitRegime": "TREND" if features.get('regime', 0) == 1 else "RANGE",
        "exitRegimeConfidence": 0.7,
        "exitPrice": features.get('price', 0) + (entry.get('pnl', 0) / 50),  # Estimate exit price
        "volatilityAtExit": features.get('volatility', 0.0001),
        
        # ADDITIONAL METRICS
        "maxFavorablePrice": features.get('price', 0) + abs(features.get('atr', 2.0)),
        "maxAdversePrice": features.get('price', 0) - abs(features.get('atr', 2.0)),
        "stopModificationCount": 0,
        "breakevenActivated": False,
        "trailingStopActive": False,
        "regimeChangeCount": 0
    }
    
    return experience, dt

def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    experiences_dir = data_dir / 'experiences'
    
    # Create experiences directory
    experiences_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created/verified directory: {experiences_dir}")
    
    # Load unified brain training data
    unified_data_path = data_dir / 'unified_brain_training_data.json'
    
    if not unified_data_path.exists():
        print(f"❌ File not found: {unified_data_path}")
        return
    
    print(f"📖 Loading: {unified_data_path}")
    with open(unified_data_path, 'r') as f:
        unified_data = json.load(f)
    
    print(f"✅ Loaded {len(unified_data)} training entries")
    
    # Convert each entry
    converted_count = 0
    for i, entry in enumerate(unified_data):
        try:
            experience, dt = convert_to_trading_experience(entry, i)
            
            # Generate filename: yyyy-MM-dd_HHmmss_<uuid>.json
            timestamp_str = dt.strftime('%Y-%m-%d_%H%M%S')
            filename = f"{timestamp_str}_{experience['experienceId']}.json"
            filepath = experiences_dir / filename
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(experience, f, indent=2)
            
            converted_count += 1
            
            if (converted_count % 100 == 0):
                print(f"  ... converted {converted_count} experiences")
                
        except Exception as e:
            print(f"⚠️  Failed to convert entry {i}: {e}")
            continue
    
    print(f"\n✅ Successfully converted {converted_count}/{len(unified_data)} experiences")
    print(f"📁 Saved to: {experiences_dir}")
    
    # Verify
    experience_files = list(experiences_dir.glob('*.json'))
    print(f"📊 Total experience files: {len(experience_files)}")

if __name__ == '__main__':
    main()
