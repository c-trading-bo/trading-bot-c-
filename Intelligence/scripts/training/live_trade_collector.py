#!/usr/bin/env python3
"""
LIVE TRADE DATA COLLECTOR
Collects training data from EVERY trade in real-time for continuous RL improvement.
Integrates with C# trading bot to capture features at trade time.
"""

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradeCollector:
    """Collect training data from EVERY trade in real-time"""
    
    def __init__(self, output_dir="data/rl_training/live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = []
        self.trade_counter = 0
        
        logger.info(f"[COLLECTOR] Initialized - saving to {self.output_dir}")
        
    def collect_trade(self, trade_data):
        """Called after EVERY trade execution"""
        
        self.trade_counter += 1
        
        # Ensure required fields exist
        sample = {
            'timestamp': datetime.utcnow().isoformat(),
            'trade_id': trade_data.get('trade_id', f'trade_{self.trade_counter}'),
            'symbol': trade_data.get('symbol', 'ES'),
            'action': trade_data.get('action', 'HOLD'),
            'price': float(trade_data.get('price', 0)),
            'size': float(trade_data.get('size', 0)),
            'features': trade_data.get('features', []),  # 43 features at trade time
            'strategy_used': trade_data.get('strategy', 'Unknown'),
            'stop_loss': float(trade_data.get('stop_loss', 0)),
            'take_profit': float(trade_data.get('take_profit', 0)),
            'session': trade_data.get('session', 'RTH'),
            'regime': trade_data.get('regime', 'Range'),
            'atr': float(trade_data.get('atr', 1.0)),
            'rsi': float(trade_data.get('rsi', 50.0)),
            'result': None,  # Will be filled when trade closes
            'R_multiple': None,  # Will be calculated when trade closes
            'slip_ticks': trade_data.get('slip_ticks', 0.1)
        }
        
        self.current_session.append(sample)
        
        # Save immediately for real-time data
        self.save_sample(sample)
        
        logger.info(f"[COLLECTOR] Saved trade data #{self.trade_counter} - {sample['strategy_used']} {sample['action']}")
        
        return sample['trade_id']
        
    def save_sample(self, sample):
        """Save sample to JSONL file"""
        
        date_str = datetime.utcnow().strftime('%Y%m%d')
        filename = self.output_dir / f"live_trades_{date_str}.jsonl"
        
        with open(filename, 'a') as f:
            f.write(json.dumps(sample) + '\n')
    
    def update_trade_result(self, trade_id, result_data):
        """Update trade with final result for training"""
        
        # Find trade and update
        for sample in self.current_session:
            if sample.get('trade_id') == trade_id:
                sample['result'] = result_data.get('result', 'unknown')  # 'win' or 'loss'
                sample['pnl'] = float(result_data.get('pnl', 0))
                sample['exit_price'] = float(result_data.get('exit_price', 0))
                sample['exit_time'] = result_data.get('exit_time', datetime.utcnow().isoformat())
                sample['holding_time_minutes'] = float(result_data.get('holding_time_minutes', 0))
                sample['R_multiple'] = float(result_data.get('R_multiple', 0))
                sample['max_drawdown'] = float(result_data.get('max_drawdown', 0))
                
                # This becomes a complete training sample
                self.save_complete_sample(sample)
                
                logger.info(f"[TRAINING] Updated trade {trade_id}: {sample['result']} (R={sample['R_multiple']:.2f})")
                break
    
    def save_complete_sample(self, sample):
        """Save completed trade as training data"""
        
        # Save to completed trades file
        completed_file = self.output_dir / "completed_trades.jsonl"
        with open(completed_file, 'a') as f:
            f.write(json.dumps(sample) + '\n')
        
        # Also save to strategy-specific file
        strategy = sample.get('strategy_used', 'unknown').lower()
        strategy_file = self.output_dir / f"completed_{strategy}_trades.jsonl"
        with open(strategy_file, 'a') as f:
            f.write(json.dumps(sample) + '\n')
        
        logger.info(f"[TRAINING] New training sample added: {sample['result']} for {sample['strategy_used']}")
    
    def get_session_stats(self):
        """Get current session statistics"""
        
        if not self.current_session:
            return {"trades": 0, "completed": 0}
        
        completed = len([s for s in self.current_session if s.get('result') is not None])
        
        return {
            "trades": len(self.current_session),
            "completed": completed,
            "pending": len(self.current_session) - completed,
            "win_rate": sum(1 for s in self.current_session if s.get('result') == 'win') / max(completed, 1),
            "avg_r_multiple": sum(s.get('R_multiple', 0) for s in self.current_session if s.get('R_multiple')) / max(completed, 1)
        }
    
    def export_training_batch(self, min_samples=50):
        """Export completed trades for batch training"""
        
        completed_file = self.output_dir / "completed_trades.jsonl"
        
        if not completed_file.exists():
            logger.warning("[EXPORT] No completed trades file found")
            return None
        
        # Count samples
        with open(completed_file, 'r') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        if len(samples) < min_samples:
            logger.info(f"[EXPORT] Need {min_samples} samples, have {len(samples)} - waiting for more")
            return None
        
        # Export to CSV for RL training
        import pandas as pd
        
        csv_data = []
        for sample in samples:
            if sample.get('R_multiple') is not None:  # Only completed trades
                features = sample.get('features', [])
                if len(features) >= 20:  # Ensure we have enough features
                    csv_row = {
                        'timestamp': sample['timestamp'],
                        'symbol': sample['symbol'],
                        'session': sample['session'],
                        'regime': sample['regime'],
                        'R_multiple': sample['R_multiple'],
                        'slip_ticks': sample.get('slip_ticks', 0.1),
                        **{f'feature_{i+1}': features[i] for i in range(min(20, len(features)))}
                    }
                    csv_data.append(csv_row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            export_file = self.output_dir / f"training_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(export_file, index=False)
            
            logger.info(f"[EXPORT] Exported {len(csv_data)} training samples to {export_file}")
            return str(export_file)
        
        return None
    
    def cleanup_old_files(self, days=7):
        """Clean up old training files"""
        
        import time
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        for file_path in self.output_dir.glob("live_trades_*.jsonl"):
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                logger.info(f"[CLEANUP] Removed old file: {file_path}")

# C# Integration Helper Functions
def create_trade_data_from_csharp(signal_data):
    """Convert C# signal data to trade data format"""
    
    return {
        'trade_id': signal_data.get('Id'),
        'symbol': signal_data.get('Symbol'),
        'action': signal_data.get('Direction'),  # BUY/SELL
        'price': signal_data.get('Entry'),
        'size': signal_data.get('Size'),
        'strategy': signal_data.get('Strategy'),
        'stop_loss': signal_data.get('StopLoss'),
        'take_profit': signal_data.get('TakeProfit'),
        'features': signal_data.get('Features', []),
        'session': signal_data.get('Session', 'RTH'),
        'regime': signal_data.get('Regime', 'Range'),
        'atr': signal_data.get('Atr', 1.0),
        'rsi': signal_data.get('Rsi', 50.0),
        'slip_ticks': signal_data.get('SlipTicks', 0.1)
    }

def create_result_data_from_csharp(outcome_data):
    """Convert C# outcome data to result data format"""
    
    return {
        'result': 'win' if outcome_data.get('IsWin', False) else 'loss',
        'pnl': outcome_data.get('ActualPnl', 0),
        'exit_price': outcome_data.get('ExitPrice', 0),
        'exit_time': outcome_data.get('ExitTime', datetime.utcnow().isoformat()),
        'holding_time_minutes': outcome_data.get('HoldingTimeMinutes', 0),
        'R_multiple': outcome_data.get('ActualRMultiple', 0),
        'max_drawdown': outcome_data.get('MaxDrawdown', 0)
    }

# Global collector instance for C# integration
_global_collector = None

def get_global_collector():
    """Get or create global collector instance"""
    global _global_collector
    if _global_collector is None:
        _global_collector = LiveTradeCollector()
    return _global_collector

def collect_trade_from_csharp(signal_json):
    """Entry point for C# to collect trade data"""
    try:
        signal_data = json.loads(signal_json) if isinstance(signal_json, str) else signal_json
        trade_data = create_trade_data_from_csharp(signal_data)
        
        collector = get_global_collector()
        trade_id = collector.collect_trade(trade_data)
        
        return trade_id
    except Exception as e:
        logger.error(f"[ERROR] Failed to collect trade from C#: {e}")
        return None

def update_trade_result_from_csharp(trade_id, outcome_json):
    """Entry point for C# to update trade results"""
    try:
        outcome_data = json.loads(outcome_json) if isinstance(outcome_json, str) else outcome_json
        result_data = create_result_data_from_csharp(outcome_data)
        
        collector = get_global_collector()
        collector.update_trade_result(trade_id, result_data)
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to update trade result from C#: {e}")
        return False

if __name__ == "__main__":
    # Test the collector
    print("=" * 50)
    print("LIVE TRADE COLLECTOR TEST")
    print("=" * 50)
    
    collector = LiveTradeCollector("test_data")
    
    # Test trade collection
    test_trade = {
        'symbol': 'ES',
        'action': 'BUY',
        'price': 4500.25,
        'size': 1.0,
        'strategy': 'MeanReversion',
        'features': list(range(43)),  # 43 test features
        'stop_loss': 4495.0,
        'take_profit': 4510.0
    }
    
    trade_id = collector.collect_trade(test_trade)
    print(f"Collected trade: {trade_id}")
    
    # Test result update
    test_result = {
        'result': 'win',
        'pnl': 125.0,
        'exit_price': 4505.25,
        'R_multiple': 1.25,
        'holding_time_minutes': 15
    }
    
    collector.update_trade_result(trade_id, test_result)
    
    # Print stats
    stats = collector.get_session_stats()
    print(f"Session stats: {stats}")
    
    print("[SUCCESS] Live trade collector is working!")