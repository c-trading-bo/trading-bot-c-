#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Training for Position Sizing
Implements requirement 7: Deploy Online Learning + SAC (/ml_online/)
"""

import os
import logging
import numpy as np
import torch
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import yaml
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingSACEnvironment(gym.Env):
    """Custom trading environment for SAC position sizing"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Environment configuration
        self.config = config
        self.max_position = config.get('max_position', 5)
        self.lookback = config.get('lookback', 20)
        
        # Action space: position size (-max_position to +max_position)
        self.action_space = gym.spaces.Box(
            low=-self.max_position, 
            high=self.max_position, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: price features, position, P&L, etc.
        obs_dim = self.lookback * 4 + 3  # OHLC + position + pnl + time
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # State variables
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.entry_price = 0.0
        
        # Generate synthetic price data for demo
        self.prices = self._generate_price_data()
        
        return self._get_observation()
        
    def step(self, action):
        """Execute one step in the environment"""
        # Clip action to valid range
        target_position = np.clip(action[0], -self.max_position, self.max_position)
        
        # Calculate current price
        current_price = self.prices[self.current_step]
        
        # Calculate P&L from position change
        if self.position != 0:
            price_change = current_price - self.entry_price
            realized_pnl = self.position * price_change
            self.total_pnl += realized_pnl
            
        # Update position
        position_change = target_position - self.position
        self.position = target_position
        
        if abs(self.position) > 0.01:  # Have position
            self.entry_price = current_price
            
        # Calculate reward (risk-adjusted returns)
        reward = self._calculate_reward(position_change, current_price)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Update unrealized P&L
        if abs(self.position) > 0.01:
            self.unrealized_pnl = self.position * (current_price - self.entry_price)
        else:
            self.unrealized_pnl = 0.0
            
        obs = self._get_observation()
        info = {
            'position': self.position,
            'pnl': self.total_pnl + self.unrealized_pnl,
            'price': current_price
        }
        
        return obs, reward, done, info
        
    def _generate_price_data(self, length: int = 1000) -> np.ndarray:
        """Generate synthetic price data for training"""
        # Simple random walk with trend
        returns = np.random.normal(0.0001, 0.01, length)
        prices = 4500 + np.cumsum(returns * 4500)  # ES-like prices
        return prices
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        # Price features (OHLC over lookback window)
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step + 1
        
        price_window = self.prices[start_idx:end_idx]
        
        # Pad if needed
        if len(price_window) < self.lookback:
            padding = np.full(self.lookback - len(price_window), price_window[0])
            price_window = np.concatenate([padding, price_window])
            
        # Create OHLC from prices (simplified)
        ohlc = np.column_stack([
            price_window,  # Open (simplified)
            price_window,  # High (simplified)
            price_window,  # Low (simplified)
            price_window   # Close
        ]).flatten()
        
        # Add position and P&L info
        state_info = np.array([
            self.position / self.max_position,  # Normalized position
            self.total_pnl / 1000.0,           # Normalized total P&L
            self.current_step / len(self.prices)  # Time progress
        ])
        
        return np.concatenate([ohlc, state_info]).astype(np.float32)
        
    def _calculate_reward(self, position_change: float, current_price: float) -> float:
        """Calculate reward for RL training"""
        # Base reward: P&L
        pnl_reward = self.total_pnl / 1000.0
        
        # Penalty for large position changes (transaction costs)
        transaction_penalty = -abs(position_change) * 0.01
        
        # Penalty for excessive positions
        position_penalty = -abs(self.position) * 0.001 if abs(self.position) > 3 else 0
        
        # Risk-adjusted reward (Sharpe-like)
        total_reward = pnl_reward + transaction_penalty + position_penalty
        
        return total_reward

class SACTrainer:
    """SAC trainer for position sizing"""
    
    def __init__(self, config_path: str = 'ml_online/config_online.yaml'):
        self.config = self.load_config(config_path)
        self.models_path = Path('data/rl/sac')
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> dict:
        """Load SAC training configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('sac', {})
        except FileNotFoundError:
            # Default SAC config
            return {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 256,
                'train_timesteps': 100000,
                'eval_episodes': 10
            }
    
    def create_environment(self) -> gym.Env:
        """Create training environment"""
        env_config = {
            'max_position': 5,
            'lookback': 20
        }
        return TradingSACEnvironment(env_config)
    
    def train(self) -> str:
        """Train SAC model"""
        logger.info("[SAC_TRAINING] Starting SAC training for position sizing")
        
        # Create environment
        env = Monitor(self.create_environment())
        
        # Create SAC model
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.config.get('learning_rate', 0.0003),
            gamma=self.config.get('gamma', 0.99),
            tau=self.config.get('tau', 0.005),
            batch_size=self.config.get('batch_size', 256),
            verbose=1,
            tensorboard_log="./sac_tensorboard/"
        )
        
        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.models_path),
            name_prefix="sac_checkpoint"
        )
        
        eval_env = Monitor(self.create_environment())
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.models_path),
            log_path=str(self.models_path),
            eval_freq=5000,
            n_eval_episodes=self.config.get('eval_episodes', 10),
            deterministic=True,
            render=False
        )
        
        # Train the model
        timesteps = self.config.get('train_timesteps', 100000)
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"sac_policy_{timestamp}"
        model_path = self.models_path / f"{model_filename}.zip"
        
        model.save(str(model_path))
        
        # Create latest symlink
        latest_path = self.models_path / "sac_policy_latest.zip"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path.name)
        
        logger.info(f"[SAC_TRAINING] ✅ SAC training completed: {model_path}")
        return str(model_path)
    
    def evaluate(self, model_path: str, episodes: int = 10) -> dict:
        """Evaluate trained SAC model"""
        logger.info(f"[SAC_EVAL] Evaluating SAC model: {model_path}")
        
        # Load model
        model = SAC.load(model_path)
        
        # Create evaluation environment
        env = self.create_environment()
        
        total_rewards = []
        total_pnls = []
        
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            total_pnls.append(info['pnl'])
            
        results = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_pnl': np.mean(total_pnls),
            'std_pnl': np.std(total_pnls),
            'sharpe_ratio': np.mean(total_rewards) / (np.std(total_rewards) + 1e-8)
        }
        
        logger.info(f"[SAC_EVAL] Results: Reward={results['mean_reward']:.3f}±{results['std_reward']:.3f}, PnL={results['mean_pnl']:.2f}±{results['std_pnl']:.2f}")
        return results

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC Training for Position Sizing')
    parser.add_argument('--config', default='ml_online/config_online.yaml',
                       help='Configuration file path')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model-path', help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    trainer = SACTrainer(args.config)
    
    if args.eval_only:
        if not args.model_path:
            args.model_path = "data/rl/sac/sac_policy_latest.zip"
        results = trainer.evaluate(args.model_path)
        print(f"Evaluation results: {results}")
    else:
        model_path = trainer.train()
        print(f"Training completed: {model_path}")
        
        # Evaluate the trained model
        results = trainer.evaluate(model_path)
        print(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()