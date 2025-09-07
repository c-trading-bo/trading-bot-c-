import os, sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from neural_ucb_topstep import NeuralUCBTopStep

def train_ucb():
    """
    Train Neural UCB model with NaN guards and proper data validation
    """
    print("🧠 Starting Neural UCB training with NaN protection...")
    
    # Load your ES/NQ data
    data_path = os.getenv('UCB_DATA_PATH', 'es_nq_1min.csv')
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("📝 Please provide ES/NQ 1-minute data with columns:")
        print("   ES_price, NQ_price, ES_volume, NQ_volume, VIX, TICK, ADD, RSI_ES, RSI_NQ")
        return
    
    try:
        data = pd.read_csv(data_path)
        print(f"📊 Loaded {len(data)} rows from {data_path}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    features, rewards = [], []
    skipped_nan = 0

    # Enhanced feature engineering with NaN protection
    print("🔧 Engineering features with NaN guards...")
    
    for i in range(100, len(data) - 6):
        try:
            # Basic price/volume features with NaN protection
            es_price = float(data.get('ES_price', pd.Series([0]*len(data)))[i])
            nq_price = float(data.get('NQ_price', pd.Series([0]*len(data)))[i])
            es_volume = float(data.get('ES_volume', pd.Series([0]*len(data)))[i])
            nq_volume = float(data.get('NQ_volume', pd.Series([0]*len(data)))[i])
            
            # Market indicators with fallbacks
            vix = float(data.get('VIX', pd.Series([15]*len(data)))[i])  # Default VIX = 15
            tick = float(data.get('TICK', pd.Series([0]*len(data)))[i])
            add = float(data.get('ADD', pd.Series([0]*len(data)))[i])
            rsi_es = float(data.get('RSI_ES', pd.Series([50]*len(data)))[i])  # Default RSI = 50
            rsi_nq = float(data.get('RSI_NQ', pd.Series([50]*len(data)))[i])
            
            # Correlation calculation with NaN guard
            try:
                es_window = data['ES_price'][i-50:i].values
                nq_window = data['NQ_price'][i-50:i].values
                
                # Check for valid data
                if len(es_window) < 10 or len(nq_window) < 10:
                    correlation = 0.0
                elif np.std(es_window) < 1e-8 or np.std(nq_window) < 1e-8:
                    correlation = 0.0  # Flat window
                else:
                    corr_matrix = np.corrcoef(es_window, nq_window)
                    correlation = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
                    
                # Critical NaN guard
                correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
                
            except Exception:
                correlation = 0.0
            
            # Build feature vector
            feat = [
                es_price, nq_price, es_volume, nq_volume, vix, tick, add, 
                correlation, rsi_es, rsi_nq
            ]
            
            # Apply NaN protection to all features
            feat = [np.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6) for f in feat]
            
            # Pad to 50 dimensions
            if len(feat) < 50: 
                feat += [0.0] * (50 - len(feat))
            features.append(feat[:50])

            # Future return calculation with protection
            try:
                current_price = es_price
                future_price = float(data['ES_price'][i+5])
                
                if current_price > 1e-8:  # Avoid division by zero
                    future_return = (future_price - current_price) / current_price
                    future_return = np.nan_to_num(future_return, nan=0.0, posinf=0.1, neginf=-0.1)
                    future_return = np.clip(future_return, -0.1, 0.1)  # Clip extreme returns
                else:
                    future_return = 0.0
                    
                rewards.append(future_return)
                
            except Exception:
                rewards.append(0.0)
                
        except Exception as e:
            skipped_nan += 1
            continue

    print(f"✅ Processed {len(features)} samples, skipped {skipped_nan} NaN rows")
    
    if len(features) < 100:
        print("❌ Insufficient training data after cleaning")
        return

    # Convert to tensors with final NaN check
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(rewards, dtype=np.float32)
    
    # Final NaN protection
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=0.1, neginf=-0.1)
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1)
    
    print(f"📊 Training data shape: X={X_tensor.shape}, y={y_tensor.shape}")
    print(f"📊 Feature stats: mean={X_tensor.mean():.4f}, std={X_tensor.std():.4f}")
    print(f"📊 Return stats: mean={y_tensor.mean():.6f}, std={y_tensor.std():.6f}")

    # Create dataset and dataloader
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    # Initialize model
    model = NeuralUCBTopStep()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)

    print("🎯 Starting training...")
    
    # Training loop with NaN monitoring
    for epoch in range(25):
        epoch_loss = 0.0
        nan_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            # Check for NaNs in batch
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                nan_batches += 1
                continue
                
            mu = model.value_network(batch_x)
            sigma = model.uncertainty_network(batch_x) + 1e-6
            
            # Heteroscedastic negative log-likelihood
            nll = 0.5 * torch.log(sigma) + 0.5 * ((batch_y - mu) ** 2) / sigma
            loss = nll.mean()
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️  NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / max(len(loader) - nan_batches, 1)
        scheduler.step()
        
        if nan_batches > 0:
            print(f"Epoch {epoch+1:2d}/25 | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | NaN batches: {nan_batches}")
        else:
            print(f"Epoch {epoch+1:2d}/25 | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "neural_ucb_topstep.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")
    
    # Validation
    model.eval()
    print("\n🧪 Running validation...")
    
    with torch.no_grad():
        test_x = X_tensor[:100]  # First 100 samples
        mu = model.value_network(test_x)
        sigma = model.uncertainty_network(test_x)
        
        print(f"📊 Validation - Value range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"📊 Validation - Uncertainty range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        
        # Test recommendation
        ucb_integration = sys.modules[__name__]
        if hasattr(ucb_integration, 'UCBIntegration'):
            test_ucb = UCBIntegration(weights_path=save_path)
            test_market = {
                "es_price": 5300.0, "nq_price": 19000.0, "es_volume": 100000, "nq_volume": 80000,
                "es_atr": 10.0, "nq_atr": 25.0, "vix": 15.0, "tick": 200, "add": 500,
                "correlation": 0.8, "rsi_es": 55, "rsi_nq": 60, "instrument": "ES"
            }
            rec = test_ucb.get_recommendation(json.dumps(test_market))
            print(f"🧪 Test recommendation: {rec}")

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    train_ucb()
