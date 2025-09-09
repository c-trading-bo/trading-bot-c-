#!/usr/bin/env python3
"""
Monthly Training Pipeline with Walk-Forward Analysis
Merged functionality from BacktestHarnessService.cs
Implements rolling-window backtest harness with purge/embargo logic and auto-retrain
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestWindow:
    """Backtest window configuration"""
    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime
    purge_duration: timedelta = timedelta(days=1)
    embargo_duration: timedelta = timedelta(days=1)

@dataclass
class BacktestResult:
    """Backtest result metrics"""
    model_name: str
    window_id: str
    training_period: str
    test_period: str
    sharpe_ratio: float
    returns: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    performance_decay: float
    needs_retrain: bool

@dataclass
class RetrainingTask:
    """Model retraining task"""
    model_name: str
    training_start: datetime
    training_end: datetime
    trigger_reason: str
    status: str = "Pending"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class WalkForwardBacktester:
    """
    Rolling-window backtest harness with purge/embargo logic and auto-retrain
    Merged from UnifiedOrchestrator.Services.BacktestHarnessService
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.results_dir = Path(config.get('results_dir', 'results/backtests'))
        self.retrain_dir = Path(config.get('retrain_dir', 'data/retrain_tasks'))
        
        # Backtest configuration
        self.window_size_months = config.get('window_size_months', 3)
        self.step_size_months = config.get('step_size_months', 1)
        self.min_performance_threshold = config.get('min_performance_threshold', 0.1)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.15)
        self.performance_decay_threshold = config.get('performance_decay_threshold', 0.3)
        
        # Create directories
        for directory in [self.models_dir, self.data_dir, self.results_dir, self.retrain_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Walk-Forward Backtester initialized")
        logger.info(f"   Window: {self.window_size_months}M, Step: {self.step_size_months}M")
        logger.info(f"   Performance threshold: {self.min_performance_threshold}")
        logger.info(f"   Max drawdown: {self.max_drawdown_threshold}")

    async def run_walk_forward_analysis(self, 
                                       model_names: List[str],
                                       start_date: datetime,
                                       end_date: datetime) -> List[BacktestResult]:
        """
        Run walk-forward analysis for specified models
        """
        logger.info(f"🔄 Starting walk-forward analysis: {start_date} to {end_date}")
        
        # Generate windows
        windows = self.generate_windows(start_date, end_date)
        logger.info(f"📅 Generated {len(windows)} backtest windows")
        
        all_results = []
        
        for model_name in model_names:
            logger.info(f"🤖 Processing model: {model_name}")
            model_results = []
            
            for i, window in enumerate(windows):
                logger.info(f"   Window {i+1}/{len(windows)}: {window.test_start} to {window.test_end}")
                
                try:
                    # Prepare model for window
                    model_path = await self.prepare_model_for_window(model_name, window)
                    
                    # Load market data for testing
                    test_data = await self.load_market_data(window.test_start, window.test_end)
                    
                    if test_data.empty:
                        logger.warning(f"⚠️ No test data for window {i+1}")
                        continue
                    
                    # Run backtest
                    result = await self.run_backtest(model_name, model_path, test_data, window, f"W{i+1}")
                    
                    # Check for performance decay
                    if len(model_results) > 0:
                        result.performance_decay = self.calculate_performance_decay(model_results, result)
                        result.needs_retrain = result.performance_decay > self.performance_decay_threshold
                        
                        if result.needs_retrain:
                            await self.trigger_retraining(model_name, window)
                    
                    model_results.append(result)
                    all_results.append(result)
                    
                    # Save interim results
                    await self.save_result(result)
                    
                except Exception as e:
                    logger.error(f"❌ Error in window {i+1} for {model_name}: {e}")
                    continue
            
            # Generate model summary
            await self.generate_model_summary(model_name, model_results)
        
        # Generate final report
        await self.generate_final_report(all_results)
        
        logger.info(f"✅ Walk-forward analysis completed: {len(all_results)} results")
        return all_results

    def generate_windows(self, start_date: datetime, end_date: datetime) -> List[BacktestWindow]:
        """Generate backtest windows with purge/embargo"""
        windows = []
        current_date = start_date
        
        while current_date < end_date:
            # Training period
            training_start = current_date
            training_end = current_date + timedelta(days=30 * self.window_size_months)
            
            # Purge period
            purge_start = training_end
            purge_end = purge_start + timedelta(days=1)
            
            # Test period  
            test_start = purge_end
            test_end = test_start + timedelta(days=30 * self.step_size_months)
            
            # Embargo period
            embargo_end = test_end + timedelta(days=1)
            
            if test_end <= end_date:
                window = BacktestWindow(
                    training_start=training_start,
                    training_end=training_end,
                    test_start=test_start,
                    test_end=test_end,
                    purge_duration=timedelta(days=1),
                    embargo_duration=timedelta(days=1)
                )
                windows.append(window)
            
            # Move to next window
            current_date += timedelta(days=30 * self.step_size_months)
        
        return windows

    async def prepare_model_for_window(self, model_name: str, window: BacktestWindow) -> str:
        """Prepare model for specific window"""
        try:
            # Look for model file for this training period
            model_pattern = f"{model_name}_{window.training_start.strftime('%Y%m%d')}*.onnx"
            model_files = list(self.models_dir.glob(model_pattern))
            
            if model_files:
                model_path = str(model_files[0])
                logger.debug(f"📁 Using existing model: {model_path}")
                return model_path
            
            # Fallback to latest model
            latest_pattern = f"{model_name}_latest.onnx"
            latest_files = list(self.models_dir.glob(latest_pattern))
            
            if latest_files:
                model_path = str(latest_files[0])
                logger.debug(f"📁 Using latest model: {model_path}")
                return model_path
            
            # Last resort - any model with this name
            any_pattern = f"{model_name}*.onnx"
            any_files = list(self.models_dir.glob(any_pattern))
            
            if any_files:
                model_path = str(any_files[0])
                logger.debug(f"📁 Using fallback model: {model_path}")
                return model_path
            
            raise FileNotFoundError(f"No model found for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare model for window: {e}")
            raise

    async def load_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load market data for backtesting"""
        try:
            # Look for data files
            data_files = []
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                data_file = self.data_dir / f"market_data_{date_str}.parquet"
                
                if data_file.exists():
                    data_files.append(data_file)
                
                current_date += timedelta(days=1)
            
            if not data_files:
                logger.warning(f"⚠️ No market data files found for {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Load and concatenate data
            dataframes = []
            for data_file in data_files:
                try:
                    df = pd.read_parquet(data_file)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {data_file}: {e}")
            
            if not dataframes:
                return pd.DataFrame()
            
            combined_data = pd.concat(dataframes, ignore_index=True)
            logger.debug(f"📊 Loaded {len(combined_data)} market data rows")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load market data: {e}")
            return pd.DataFrame()

    async def run_backtest(self, 
                          model_name: str,
                          model_path: str, 
                          test_data: pd.DataFrame,
                          window: BacktestWindow,
                          window_id: str) -> BacktestResult:
        """Run backtest for a specific window"""
        try:
            # Simulate backtest (in real implementation, would run actual backtest)
            # This is a placeholder implementation
            
            # Calculate basic metrics
            returns = np.random.normal(0.001, 0.02, len(test_data))  # Simulated returns
            cumulative_returns = np.cumprod(1 + returns) - 1
            
            # Calculate metrics
            total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Drawdown calculation
            peak = np.maximum.accumulate(cumulative_returns + 1)
            drawdown = (cumulative_returns + 1) / peak - 1
            max_drawdown = np.min(drawdown)
            
            # Trade statistics (simulated)
            total_trades = len(test_data) // 10  # Assume 1 trade per 10 bars
            win_rate = 0.55 + np.random.normal(0, 0.1)  # Simulated win rate
            avg_trade_duration = 24 * 3  # Average 3 days
            
            result = BacktestResult(
                model_name=model_name,
                window_id=window_id,
                training_period=f"{window.training_start.date()} to {window.training_end.date()}",
                test_period=f"{window.test_start.date()} to {window.test_end.date()}",
                sharpe_ratio=sharpe_ratio,
                returns=total_return,
                max_drawdown=abs(max_drawdown),
                win_rate=max(0, min(1, win_rate)),
                avg_trade_duration=avg_trade_duration,
                total_trades=total_trades,
                performance_decay=0.0,
                needs_retrain=False
            )
            
            logger.info(f"📈 Backtest result: Sharpe={sharpe_ratio:.3f}, Return={total_return:.3f}, DD={abs(max_drawdown):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise

    def calculate_performance_decay(self, 
                                   previous_results: List[BacktestResult], 
                                   current_result: BacktestResult) -> float:
        """Calculate performance decay compared to recent windows"""
        if len(previous_results) < 3:
            return 0.0
        
        # Use last 3 windows for comparison
        recent_results = previous_results[-3:]
        recent_avg_sharpe = np.mean([r.sharpe_ratio for r in recent_results])
        
        if recent_avg_sharpe <= 0:
            return 0.0
        
        decay = (recent_avg_sharpe - current_result.sharpe_ratio) / recent_avg_sharpe
        return max(0, decay)

    async def trigger_retraining(self, model_name: str, window: BacktestWindow):
        """Trigger model retraining"""
        try:
            retrain_task = RetrainingTask(
                model_name=model_name,
                training_start=window.training_start,
                training_end=window.test_end,  # Include recent data
                trigger_reason="Performance decay detected"
            )
            
            # Save retraining task
            await self.save_retraining_task(retrain_task)
            
            logger.info(f"📝 Retraining task created for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger retraining: {e}")

    async def save_retraining_task(self, task: RetrainingTask):
        """Save retraining task to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retrain_{task.model_name}_{timestamp}.json"
            filepath = self.retrain_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(task), f, indent=2, default=str)
            
            logger.debug(f"💾 Saved retraining task: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save retraining task: {e}")

    async def save_result(self, result: BacktestResult):
        """Save backtest result"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.model_name}_{result.window_id}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            logger.debug(f"💾 Saved result: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save result: {e}")

    async def generate_model_summary(self, model_name: str, results: List[BacktestResult]):
        """Generate summary report for a model"""
        try:
            if not results:
                return
            
            summary = {
                "model_name": model_name,
                "total_windows": len(results),
                "avg_sharpe": np.mean([r.sharpe_ratio for r in results]),
                "avg_returns": np.mean([r.returns for r in results]),
                "avg_max_drawdown": np.mean([r.max_drawdown for r in results]),
                "avg_win_rate": np.mean([r.win_rate for r in results]),
                "retrains_triggered": sum(1 for r in results if r.needs_retrain),
                "performance_trend": "improving" if len(results) >= 2 and results[-1].sharpe_ratio > results[0].sharpe_ratio else "declining",
                "generated_at": datetime.now().isoformat()
            }
            
            filename = f"summary_{model_name}_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📋 Model summary saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate model summary: {e}")

    async def generate_final_report(self, all_results: List[BacktestResult]):
        """Generate final walk-forward analysis report"""
        try:
            if not all_results:
                return
            
            models = list(set(r.model_name for r in all_results))
            
            report = {
                "analysis_completed_at": datetime.now().isoformat(),
                "total_models": len(models),
                "total_windows": len(all_results),
                "models": {}
            }
            
            for model in models:
                model_results = [r for r in all_results if r.model_name == model]
                
                report["models"][model] = {
                    "windows_tested": len(model_results),
                    "avg_sharpe": np.mean([r.sharpe_ratio for r in model_results]),
                    "avg_returns": np.mean([r.returns for r in model_results]),
                    "max_drawdown": max([r.max_drawdown for r in model_results]),
                    "retrains_needed": sum(1 for r in model_results if r.needs_retrain),
                    "performance_stability": np.std([r.sharpe_ratio for r in model_results])
                }
            
            # Overall ranking
            model_rankings = []
            for model in models:
                model_results = [r for r in all_results if r.model_name == model]
                avg_sharpe = np.mean([r.sharpe_ratio for r in model_results])
                max_dd = max([r.max_drawdown for r in model_results])
                score = avg_sharpe / max(max_dd, 0.01)  # Risk-adjusted score
                model_rankings.append((model, score))
            
            model_rankings.sort(key=lambda x: x[1], reverse=True)
            report["model_rankings"] = [{"model": m, "score": s} for m, s in model_rankings]
            
            filename = f"walk_forward_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Final report saved: {filepath}")
            logger.info(f"🏆 Top model: {model_rankings[0][0]} (score: {model_rankings[0][1]:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate final report: {e}")

async def main():
    """Main entry point for monthly training pipeline"""
    try:
        # Load configuration
        config = {
            "models_dir": "models",
            "data_dir": "data",
            "results_dir": "results/backtests",
            "retrain_dir": "data/retrain_tasks",
            "window_size_months": 3,
            "step_size_months": 1,
            "min_performance_threshold": 0.1,
            "max_drawdown_threshold": 0.15,
            "performance_decay_threshold": 0.3
        }
        
        # Initialize backtester
        backtester = WalkForwardBacktester(config)
        
        # Define models to test
        models = ["neural_ucb", "transformer", "lstm", "xgboost"]
        
        # Define analysis period (last 12 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        logger.info(f"🚀 Starting monthly training pipeline")
        logger.info(f"   Models: {models}")
        logger.info(f"   Period: {start_date.date()} to {end_date.date()}")
        
        # Run walk-forward analysis
        results = await backtester.run_walk_forward_analysis(models, start_date, end_date)
        
        logger.info(f"✅ Monthly training pipeline completed successfully")
        logger.info(f"   Total results: {len(results)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Monthly training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())