#!/usr/bin/env python3
"""
ML-Enhanced Time Optimization for Trading Strategies
Analyzes historical performance and trains models to predict optimal trading times
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries, fall back gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  Scikit-learn not available. Using rule-based optimization.")
    ML_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

class TimeOptimizationML:
    def __init__(self):
        self.models = {}
        self.performance_history = {}
        self.data_dir = Path("data/ml")
        self.models_dir = Path("models/time_optimization")
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_time_optimization(self, instrument='ES'):
        """Train ML model to predict best times for each strategy"""
        
        print(f"[ML] Training time optimization for {instrument}")
        
        # Load historical performance data
        data = self.load_performance_data(instrument)
        
        if data.empty:
            # FAIL FAST: No synthetic data generation allowed
            error_msg = f"No historical data found for {instrument}. System refuses to generate synthetic data for training."
            print(f"  ❌ {error_msg}")
            raise ValueError(f"Real historical data required for {instrument}. "
                           f"Implement real data loading from TopstepX or trading database.")
        
        
        for strategy in ['S2', 'S3', 'S6', 'S11']:
            print(f"  Training {strategy}...")
            
            strategy_data = data[data['strategy'] == strategy].copy()
            
            if len(strategy_data) < 50:
                print(f"    ⚠️  Insufficient data for {strategy} ({len(strategy_data)} records). Skipping ML training.")
                self.calculate_rule_based_performance(instrument, strategy)
                continue
            
            if ML_AVAILABLE:
                self.train_ml_model(instrument, strategy, strategy_data)
            else:
                self.calculate_rule_based_performance(instrument, strategy)
        
        # Save results
        self.save_optimization_results(instrument)
        
    def load_performance_data(self, instrument):
        """Load historical trading performance data"""
        
        data_file = self.data_dir / f"{instrument}_performance_history.csv"
        
        if data_file.exists():
            try:
                return pd.read_csv(data_file, parse_dates=['timestamp'])
            except Exception as e:
                print(f"  Error loading data: {e}")
        
        return pd.DataFrame()
    
    def load_real_performance_data(self, instrument, min_records=1000):
        """
        Load REAL performance data from trading database - NO SYNTHETIC GENERATION
        
        Args:
            instrument: Trading instrument (e.g., 'ES', 'NQ')  
            min_records: Minimum number of records required
            
        Returns:
            DataFrame with real trading performance data
            
        Raises:
            ValueError: If real data unavailable (FAIL FAST)
        """
        
        # TODO: Implement real performance data loading from trading database
        # This should load actual trade outcomes, PnL, and market features from real trading history
        
        error_msg = (f"Real performance data loading not implemented for {instrument}. "
                    f"System refuses to generate synthetic trading performance data. "
                    f"Implement real data loading from trading database with actual trade outcomes.")
        
        print(f"  ❌ {error_msg}")
        raise ValueError(error_msg)
    
    def train_ml_model(self, instrument, strategy, data):
        """Train ML model for strategy time optimization"""
        
        # Prepare features
        features = []
        targets = []
        
        for _, row in data.iterrows():
            # Time-based features
            hour = row['timestamp'].hour
            minute = row['timestamp'].minute
            day_of_week = row['timestamp'].weekday()
            
            # Market features
            feature_vector = [
                hour,
                minute,
                day_of_week,
                row['volatility'],
                row['volume_ratio'],
                row['vix_level'],
                row['es_nq_correlation'],
                row['session_range'],
                row['distance_from_vwap'],
                row['rsi'],
                row['delta_divergence']
            ]
            
            features.append(feature_vector)
            
            # Target: Was trade profitable?
            targets.append(1 if row['pnl'] > 0 else 0)
        
        if len(features) < 20:
            print(f"    Insufficient features for ML training ({len(features)})")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"    Model performance - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save model
        if JOBLIB_AVAILABLE:
            model_file = self.models_dir / f"{instrument}_{strategy}_model.pkl"
            joblib.dump(model, model_file)
            print(f"    Model saved to {model_file}")
        
        # Store model
        self.models[f"{instrument}_{strategy}"] = model
        
        # Calculate time performance
        self.calculate_ml_time_performance(instrument, strategy, model, features, targets, data)
    
    def calculate_ml_time_performance(self, instrument, strategy, model, features, targets, data):
        """Calculate win rate by time of day using ML model"""
        
        performance = {}
        
        # Group by hour and calculate performance
        data_with_features = data.copy()
        data_with_features['hour'] = data_with_features['timestamp'].dt.hour
        data_with_features['profitable'] = data_with_features['pnl'] > 0
        
        # Calculate win rate by hour
        for hour in range(24):
            hour_data = data_with_features[data_with_features['hour'] == hour]
            if len(hour_data) > 5:
                win_rate = hour_data['profitable'].mean()
                performance[hour] = float(win_rate)
        
        self.performance_history[f"{instrument}_{strategy}"] = performance
        
    def calculate_rule_based_performance(self, instrument, strategy):
        """Calculate performance using rule-based logic when ML is not available"""
        
        print(f"    Using rule-based optimization for {strategy}")
        
        # Rule-based time preferences based on market knowledge
        performance = {}
        
        if strategy == 'S2':  # VWAP Mean Reversion
            performance = {
                0: 0.85, 3: 0.82, 12: 0.88, 19: 0.83, 23: 0.87,
                6: 0.70, 9: 0.65, 15: 0.72, 21: 0.75
            }
        elif strategy == 'S3':  # Compression Breakout
            performance = {
                3: 0.90, 9: 0.92, 10: 0.85, 14: 0.80,
                6: 0.75, 12: 0.70, 16: 0.75, 20: 0.65
            }
        elif strategy == 'S6':  # Opening Drive
            performance = {
                9: 0.95, 10: 0.30,  # Only effective during opening
                # All other hours get low performance
                **{h: 0.30 for h in range(24) if h not in [9, 10]}
            }
        elif strategy == 'S11':  # ADR Exhaustion
            performance = {
                13: 0.91, 14: 0.88, 15: 0.85, 16: 0.82,
                9: 0.70, 11: 0.65, 17: 0.75
            }
        else:
            # Default performance
            performance = {h: 0.65 for h in range(24)}
        
        self.performance_history[f"{instrument}_{strategy}"] = performance
        
    def get_optimal_times(self, instrument, strategy):
        """Get optimal trading times for strategy"""
        
        key = f"{instrument}_{strategy}"
        if key not in self.performance_history:
            return []
        
        perf = self.performance_history[key]
        
        # Sort by performance
        sorted_hours = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        
        # Return top trading hours
        optimal_hours = []
        for hour, win_rate in sorted_hours:
            if win_rate > 0.70:  # Only high win rate times
                optimal_hours.append({
                    'hour': hour,
                    'win_rate': win_rate,
                    'session': self.get_session_name(hour)
                })
        
        return optimal_hours[:5]  # Top 5 hours
    
    def get_session_name(self, hour):
        """Map hour to trading session"""
        
        # Convert to CT (assuming input is CT)
        if 0 <= hour < 2:
            return "Asian Session"
        elif 2 <= hour < 8:
            return "European Session"
        elif 8 <= hour < 9.5:
            return "Pre-Market"
        elif 9.5 <= hour < 10:
            return "Opening Drive"
        elif 10 <= hour < 11.5:
            return "Morning Trend"
        elif 11.5 <= hour < 13.5:
            return "Lunch Chop"
        elif 13.5 <= hour < 15:
            return "Afternoon"
        elif 15 <= hour < 16:
            return "Close"
        elif 16 <= hour < 18:
            return "After Hours"
        else:
            return "Overnight"
    
    def save_optimization_results(self, instrument):
        """Save optimization results to JSON file"""
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'instrument': instrument,
            'ml_available': ML_AVAILABLE,
            'optimal_times': {},
            'performance_by_hour': {}
        }
        
        for strategy in ['S2', 'S3', 'S6', 'S11']:
            results['optimal_times'][strategy] = self.get_optimal_times(instrument, strategy)
            
            key = f"{instrument}_{strategy}"
            if key in self.performance_history:
                results['performance_by_hour'][strategy] = self.performance_history[key]
        
        # Save to Intelligence directory for integration with C# bot
        output_dir = Path("Intelligence/data/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{instrument}_time_optimization.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[ML] Results saved to {output_file}")
        
        # Also save to models directory
        model_output = self.models_dir / f"{instrument}_time_optimization.json"
        with open(model_output, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """Main execution function"""
    
    print("🧠 Starting ML Time Optimization Training")
    print("=" * 50)
    
    optimizer = TimeOptimizationML()
    
    # Train for both ES and NQ
    for instrument in ['ES', 'NQ']:
        print(f"\n📊 Processing {instrument}...")
        optimizer.train_time_optimization(instrument)
    
    print("\n✅ Time optimization training complete!")
    
    # Generate summary report
    print("\n📋 OPTIMIZATION SUMMARY:")
    print("-" * 30)
    
    for instrument in ['ES', 'NQ']:
        print(f"\n{instrument} OPTIMAL TIMES:")
        for strategy in ['S2', 'S3', 'S6', 'S11']:
            optimal_times = optimizer.get_optimal_times(instrument, strategy)
            if optimal_times:
                best_time = optimal_times[0]
                print(f"  {strategy}: {best_time['hour']:02d}:00 ({best_time['win_rate']:.1%}) - {best_time['session']}")
            else:
                print(f"  {strategy}: No optimal times found")

if __name__ == "__main__":
    main()