import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


class SizerEnv:
    """Dataset-driven environment. One step per candidate trade.
    Action a ∈ {0.50,0.75,1.00,1.25,1.50} (pos multiplier). Reward = a * R_multiple - slipR.
    CVaR constraint enforced in PPO loss via Lagrangian penalty.
    """
    
    def __init__(self, df: pd.DataFrame, actions: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5), feature_cols: Optional[List[str]] = None):
        """Initialize the sizing environment.
        
        Args:
            df: DataFrame with candidate trade data
            actions: Available position multipliers 
            feature_cols: Feature columns to use as state. If None, auto-detect.
        """
        self.df = df.reset_index(drop=True)
        self.A = np.array(actions, dtype=np.float32)
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            excluded_cols = {
                'timestamp', 'symbol', 'session', 'regime', 
                'label_win', 'R_multiple', 'slip_ticks'
            }
            self.cols = [c for c in df.columns if c not in excluded_cols]
        else:
            self.cols = feature_cols
            
        self.i = 0
        self.n = len(self.df)
        self.done = False
        
        # Validate required columns
        required_cols = ['R_multiple']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.i = 0
        self.done = False
        return self._obs(self.i)

    def step(self, a_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return next observation, reward, done, info.
        
        Args:
            a_idx: Index into action array
            
        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")
            
        a_idx = int(a_idx)
        if a_idx < 0 or a_idx >= len(self.A):
            raise ValueError(f"Invalid action {a_idx}. Must be in range [0, {len(self.A)-1}]")
            
        a = self.A[a_idx]
        row = self.df.iloc[self.i]
        
        R = float(row['R_multiple'])
        slip_ticks = float(row.get('slip_ticks', 0.0))
        
        # Approximate slip cost in R units; adjust per symbol if needed
        slip_r = 0.1 * slip_ticks * a
        
        # Reward = position-scaled R-multiple minus slip cost
        r = a * R - slip_r
        
        self.i += 1
        if self.i >= self.n:
            self.done = True
            
        return self._obs(self.i), r, self.done, {
            'action_value': a, 
            'R_multiple': R, 
            'slip_cost_R': slip_r,
            'raw_reward': a * R,
            'slip_ticks': slip_ticks
        }

    def _obs(self, i: int) -> np.ndarray:
        """Get observation at index i."""
        if i >= self.n:
            i = self.n - 1  # Clamp to last valid index
            
        try:
            v = self.df.iloc[i][self.cols].values.astype(np.float32)
            # Handle any NaN values
            v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            return v
        except Exception:
            # Fallback: return zeros if extraction fails
            return np.zeros(len(self.cols), dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        """Observation space dimension."""
        return len(self.cols)
    
    @property
    def act_dim(self) -> int:
        """Action space dimension."""
        return len(self.A)
    
    @property
    def action_space(self) -> np.ndarray:
        """Available actions (position multipliers)."""
        return self.A.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get names of feature columns used as observations."""
        return self.cols.copy()
    
    def sample_action(self) -> int:
        """Sample random action for exploration."""
        return np.random.randint(0, self.act_dim)


# Helper function for CVaR calculation
def cvar_tail(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (CVaR) at alpha level.
    
    Args:
        returns: Array of returns/rewards
        alpha: Confidence level (0.95 = 95% CVaR)
        
    Returns:
        CVaR value (negative indicates loss)
    """
    if len(returns) == 0:
        return 0.0
        
    returns = np.asarray(returns)
    # Sort returns (ascending - worst first)
    sorted_returns = np.sort(returns)
    
    # Find (1-alpha) percentile index
    tail_index = int(np.ceil((1 - alpha) * len(returns)))
    tail_index = max(0, min(tail_index, len(returns) - 1))
    
    # CVaR is mean of worst (1-alpha)% returns
    if tail_index == 0:
        return sorted_returns[0]
    else:
        return np.mean(sorted_returns[:tail_index])


if __name__ == "__main__":
    # Test environment with dummy data
    import pandas as pd
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'symbol': ['ES'] * n_samples,
        'session': np.random.choice(['RTH', 'ETH'], n_samples),
        'regime': np.random.choice(['Range', 'Trend', 'Vol'], n_samples),
        'R_multiple': np.random.normal(0.1, 1.5, n_samples),
        'slip_ticks': np.random.exponential(0.5, n_samples),
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.uniform(-1, 1, n_samples),
        'label_win': np.random.choice([0, 1], n_samples)
    })
    
    # Test environment
    env = SizerEnv(test_data)
    print(f"Observation dim: {env.obs_dim}")
    print(f"Action dim: {env.act_dim}")
    print(f"Actions: {env.action_space}")
    print(f"Feature names: {env.get_feature_names()}")
    
    # Run a few steps
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    total_reward = 0
    for step in range(5):
        action = env.sample_action()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: action={action} (mult={info['action_value']:.2f}), "
              f"reward={reward:.3f}, R={info['R_multiple']:.2f}")
        if done:
            break
    
    print(f"Total reward: {total_reward:.3f}")
    
    # Test CVaR calculation
    test_returns = np.random.normal(-0.1, 2.0, 1000)
    cvar_95 = cvar_tail(test_returns, 0.95)
    print(f"CVaR@95%: {cvar_95:.3f}")
