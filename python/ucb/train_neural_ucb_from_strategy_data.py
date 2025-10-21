"""
Neural-UCB Training Script for Strategy Selection Learning
Reads JSON data exported from C# NeuralUcbBandit during Lab Mode
Retrains neural networks for S2/S3/S6/S11 strategy selection

Input:  models/neural_ucb_training_data.json (exported by HistoricalTrainingOrchestrator)
Output: python/ucb/ucb_state.pkl (reloaded by C# OnnxNeuralNetwork)
        models/neural_ucb_model_{arm}.onnx (per-strategy ONNX models)

Production-ready with full error handling, logging, and validation.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from neural_ucb_topstep import NeuralUCBTopStep, TopStepConfig
    TOPSTEP_AVAILABLE = True
except ImportError:
    TOPSTEP_AVAILABLE = False
    print("⚠️  Warning: neural_ucb_topstep.py not found, using standalone implementation")


class ContextVectorDecoder:
    """Decodes C# ContextVector JSON format to feature arrays"""
    
    @staticmethod
    def decode_context(context_obj: Dict, target_dim: int = 50) -> np.ndarray:
        """
        Extract features from C# ContextVector JSON structure.
        
        ContextVector format from C#:
        {
            "Features": {
                "market_regime": 0.45,
                "volatility": 0.23,
                "price_direction": 1.0,
                "time_of_day": 0.67,
                ...
            }
        }
        """
        try:
            if isinstance(context_obj, dict):
                if "Features" in context_obj:
                    features_dict = context_obj["Features"]
                    # Sort by key name for consistency
                    sorted_features = sorted(features_dict.items())
                    features = [float(v) for k, v in sorted_features]
                else:
                    # Assume all keys are features
                    sorted_features = sorted(context_obj.items())
                    features = [float(v) for k, v in sorted_features]
            elif isinstance(context_obj, list):
                features = [float(x) for x in context_obj]
            else:
                raise ValueError(f"Unexpected context format: {type(context_obj)}")
            
            # Pad or truncate to target dimension
            if len(features) < target_dim:
                features += [0.0] * (target_dim - len(features))
            elif len(features) > target_dim:
                features = features[:target_dim]
            
            # NaN protection
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Error decoding context: {e}")
            print(f"   Context object: {context_obj}")
            # Return zero vector as fallback
            return np.zeros(target_dim, dtype=np.float32)


class StrategyNeuralNetwork(nn.Module):
    """Neural network for single strategy arm (S2, S3, S6, or S11)"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        # Value prediction network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single value output (predicted reward)
        )
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (value_prediction, uncertainty)"""
        value = self.value_net(x)
        uncertainty = self.uncertainty_net(x)
        return value, uncertainty


class NeuralUCBTrainer:
    """Production-ready trainer for Neural-UCB strategy selection"""
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 NeuralUCBTrainer initialized: device={self.device}, input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        # Strategy-specific networks (one per arm)
        self.networks: Dict[str, StrategyNeuralNetwork] = {}
        self.optimizers: Dict[str, optim.Adam] = {}
    
    def load_training_data(self, json_path: str) -> Dict[str, List[Tuple[np.ndarray, float]]]:
        """
        Load training data exported from C# NeuralUcbBandit.
        
        Expected JSON format:
        {
            "S2": [
                {"context": {...}, "reward": 0.452},
                ...
            ],
            "S3": [...],
            "S6": [...],
            "S11": [...]
        }
        """
        print(f"📂 Loading training data from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Training data not found: {json_path}")
        
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        training_data: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        decoder = ContextVectorDecoder()
        
        for arm_id, samples in raw_data.items():
            if not isinstance(samples, list):
                print(f"⚠️  Skipping {arm_id}: expected list, got {type(samples)}")
                continue
            
            arm_samples = []
            skipped = 0
            
            for sample in samples:
                try:
                    # Handle tuple format: (context, reward) or dict format: {context, reward}
                    if isinstance(sample, dict):
                        context = sample.get("context") or sample.get("Context") or sample.get("Item1")
                        reward = sample.get("reward") or sample.get("Reward") or sample.get("Item2")
                    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                        context, reward = sample[0], sample[1]
                    else:
                        print(f"⚠️  Invalid sample format: {sample}")
                        skipped += 1
                        continue
                    
                    # Decode context vector
                    features = decoder.decode_context(context, self.input_dim)
                    reward_val = float(reward)
                    
                    # Validate features
                    if not np.isfinite(features).all():
                        print(f"⚠️  Non-finite features detected, skipping sample")
                        skipped += 1
                        continue
                    
                    arm_samples.append((features, reward_val))
                    
                except Exception as e:
                    print(f"⚠️  Error processing sample: {e}")
                    skipped += 1
                    continue
            
            if arm_samples:
                training_data[arm_id] = arm_samples
                print(f"✅ Loaded {len(arm_samples)} samples for {arm_id} (skipped {skipped})")
            else:
                print(f"⚠️  No valid samples for {arm_id}")
        
        return training_data
    
    def train_arm(
        self,
        arm_id: str,
        training_samples: List[Tuple[np.ndarray, float]]
    ) -> Dict[str, float]:
        """
        Train neural network for a single strategy arm.
        Returns training metrics (loss, final_reward).
        """
        print(f"\n🎯 Training {arm_id} with {len(training_samples)} samples...")
        
        # Create network and optimizer
        network = StrategyNeuralNetwork(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
        
        # Prepare data
        features = np.stack([s[0] for s in training_samples])
        rewards = np.array([s[1] for s in training_samples], dtype=np.float32)
        
        # Normalize rewards to [0, 1] range for better training
        reward_min = rewards.min()
        reward_max = rewards.max()
        reward_range = reward_max - reward_min
        
        if reward_range > 1e-6:
            rewards_normalized = (rewards - reward_min) / reward_range
        else:
            rewards_normalized = np.zeros_like(rewards)
        
        # Create DataLoader
        dataset = TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(rewards_normalized).float()
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        network.train()
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_features, batch_rewards in dataloader:
                batch_features = batch_features.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                value_pred, uncertainty_pred = network(batch_features)
                value_pred = value_pred.squeeze()
                uncertainty_pred = uncertainty_pred.squeeze()
                
                # Loss: MSE for value prediction + regularization for uncertainty
                value_loss = nn.MSELoss()(value_pred, batch_rewards)
                uncertainty_reg = uncertainty_pred.mean()  # Encourage lower uncertainty
                
                total_loss = value_loss + 0.01 * uncertainty_reg
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.6f}")
        
        # Store trained network
        network.eval()
        self.networks[arm_id] = network
        self.optimizers[arm_id] = optimizer
        
        final_loss = losses[-1] if losses else 0.0
        avg_reward = float(rewards.mean())
        
        print(f"✅ {arm_id} training complete: final_loss={final_loss:.6f}, avg_reward={avg_reward:.4f}")
        
        return {
            "final_loss": final_loss,
            "avg_reward": avg_reward,
            "sample_count": len(training_samples),
            "epochs": self.epochs
        }
    
    def train_all_arms(
        self,
        training_data: Dict[str, List[Tuple[np.ndarray, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Train all strategy arms and return metrics for each"""
        metrics = {}
        
        for arm_id, samples in training_data.items():
            if len(samples) < 10:
                print(f"⚠️  Skipping {arm_id}: insufficient samples ({len(samples)} < 10)")
                continue
            
            try:
                arm_metrics = self.train_arm(arm_id, samples)
                metrics[arm_id] = arm_metrics
            except Exception as e:
                print(f"❌ Error training {arm_id}: {e}")
                metrics[arm_id] = {"error": str(e)}
        
        return metrics
    
    def save_models(self, output_dir: str = "models") -> List[str]:
        """
        Save trained models in ONNX format for C# inference.
        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for arm_id, network in self.networks.items():
            try:
                # Export to ONNX
                onnx_path = os.path.join(output_dir, f"neural_ucb_model_{arm_id}.onnx")
                
                dummy_input = torch.randn(1, self.input_dim).to(self.device)
                
                torch.onnx.export(
                    network,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['value', 'uncertainty'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'value': {0: 'batch_size'},
                        'uncertainty': {0: 'batch_size'}
                    }
                )
                
                print(f"💾 Saved ONNX model: {onnx_path}")
                saved_files.append(onnx_path)
                
            except Exception as e:
                print(f"❌ Error exporting {arm_id} to ONNX: {e}")
        
        return saved_files
    
    def save_checkpoint(self, checkpoint_path: str = "python/ucb/ucb_state.pkl"):
        """Save training state in pickle format for Python-side persistence"""
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint = {
                "timestamp": datetime.utcnow().isoformat(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "device": self.device,
                "networks": {
                    arm_id: network.state_dict()
                    for arm_id, network in self.networks.items()
                },
                "optimizers": {
                    arm_id: optimizer.state_dict()
                    for arm_id, optimizer in self.optimizers.items()
                }
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"💾 Saved checkpoint: {checkpoint_path} ({os.path.getsize(checkpoint_path)} bytes)")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            raise


def main():
    """Main entry point for Neural-UCB training from strategy data"""
    
    parser = argparse.ArgumentParser(description="Train Neural-UCB from C# exported strategy data")
    parser.add_argument("--data-path", type=str, default="models/neural_ucb_training_data.json",
                       help="Path to JSON training data exported from C#")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save ONNX models")
    parser.add_argument("--checkpoint-path", type=str, default="python/ucb/ucb_state.pkl",
                       help="Path to save Python checkpoint")
    parser.add_argument("--input-dim", type=int, default=50,
                       help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden layer dimension")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Neural-UCB Strategy Selection Training")
    print("=" * 80)
    print(f"Data path:       {args.data_path}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Checkpoint:      {args.checkpoint_path}")
    print(f"Input dim:       {args.input_dim}")
    print(f"Hidden dim:      {args.hidden_dim}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs}")
    print("=" * 80)
    
    try:
        # Initialize trainer
        trainer = NeuralUCBTrainer(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # Load training data
        training_data = trainer.load_training_data(args.data_path)
        
        if not training_data:
            print("❌ No training data loaded. Exiting.")
            return 1
        
        print(f"\n📊 Training data summary:")
        for arm_id, samples in training_data.items():
            print(f"   {arm_id}: {len(samples)} samples")
        
        # Train all arms
        print("\n🎯 Starting training...")
        metrics = trainer.train_all_arms(training_data)
        
        # Print metrics summary
        print("\n" + "=" * 80)
        print("📈 Training Metrics Summary")
        print("=" * 80)
        for arm_id, arm_metrics in metrics.items():
            if "error" in arm_metrics:
                print(f"{arm_id}: ERROR - {arm_metrics['error']}")
            else:
                print(f"{arm_id}: loss={arm_metrics['final_loss']:.6f}, "
                      f"avg_reward={arm_metrics['avg_reward']:.4f}, "
                      f"samples={arm_metrics['sample_count']}")
        
        # Save models
        print("\n💾 Saving models...")
        saved_files = trainer.save_models(args.output_dir)
        
        # Save checkpoint
        trainer.save_checkpoint(args.checkpoint_path)
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"ONNX models saved: {len(saved_files)}")
        print(f"Checkpoint saved: {args.checkpoint_path}")
        print("Models ready for C# inference via OnnxNeuralNetwork")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON format: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
