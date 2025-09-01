# 🏠 Local vs ☁️ Cloud Architecture Guide

## 📋 Overview

Your trading bot uses a **hybrid architecture** that combines local execution for speed with cloud learning for continuous improvement. Here's exactly what runs where and why.

## 🏠 **What Stays LOCAL (Real-Time Execution)**

### ⚡ **Trading Engine & Execution**
- **Order placement and management** - Must stay local for millisecond latency
- **Risk management** - Real-time position monitoring and stops
- **Market data processing** - Live price feeds and bar aggregation  
- **Strategy signal generation** - S2, S3, S6, S11 strategy execution
- **Position tracking** - Real-time P&L and exposure monitoring

### 🧠 **Model Inference (ONNX)**
- **RL position sizing** - Uses pre-trained models locally for instant decisions
- **Strategy selection** - Meta-classifier for strategy prioritization
- **Execution quality prediction** - Risk-adjusted sizing based on market conditions

### 📊 **Data Collection**
- **Feature snapshots** - Market data at signal generation time
- **Trade outcomes** - Performance tracking for feedback loops
- **Symbol-specific data** - Both ES and NQ training data collection

**Why Local?** 
- ⚡ **Speed**: Sub-millisecond decision making
- 🔒 **Reliability**: No network dependencies for critical operations
- 💰 **Cost**: No compute charges for real-time operations

---

## ☁️ **What Runs in CLOUD (24/7 Learning)**

### 🤖 **Model Training Pipeline**
- **Feature engineering** - Advanced technical indicators and market regime detection
- **Deep learning training** - GPU-accelerated neural networks
- **Reinforcement learning** - Position sizing optimization using PPO/A2C
- **Cross-validation** - Robust model validation across time periods
- **Hyperparameter optimization** - Automated tuning for best performance

### 🔄 **Continuous Learning (Every 30 Minutes)**
- **Data merging** - Combines real trading data + vendor data + synthetic data
- **Multi-strategy training** - EmaCross, MeanReversion, Breakout, Momentum
- **Multi-symbol learning** - Both ES and NQ symbol-specific optimization
- **Model versioning** - Automatic deployment with rollback capability

### 🌐 **Infrastructure**
- **GitHub Actions** - Automated training pipeline
- **AWS S3** - Model storage and distribution
- **HMAC Security** - Cryptographic verification of model integrity
- **CDN Distribution** - Fast model downloads worldwide

**Why Cloud?**
- 🚀 **Power**: GPU acceleration for complex training
- 📈 **Scale**: Handle massive datasets
- 🔄 **Availability**: 24/7 training even when local bot is offline
- 💡 **Intelligence**: Advanced ML algorithms that require significant compute

---

## 🔄 **How They Work Together**

### **Data Flow: Local → Cloud**
1. **Local bot** generates trading signals and collects features
2. **Training data** logged to symbol-specific files (`features_es_*.jsonl`, `features_nq_*.jsonl`)
3. **Trade outcomes** recorded for performance feedback
4. **GitHub Actions** syncs data to cloud storage every 30 minutes
5. **Cloud training** processes combined datasets from all sources

### **Model Flow: Cloud → Local**
1. **Cloud training** produces optimized ONNX models
2. **Models uploaded** to S3 with cryptographic signatures
3. **Local bot** automatically downloads new models via `CloudRlTrainerEnhanced`
4. **Hot swapping** - Models updated without restarting bot
5. **Performance monitoring** - Tracks improvement from new models

---

## 🎯 **Active Learning Strategies**

Your bot is continuously learning and improving these strategies on **both ES and NQ**:

### **Strategy Types Learning 24/7:**
- ✅ **S2 (EmaCross)** - Moving average crossover with momentum confirmation
- ✅ **S3 (MeanReversion)** - Bollinger Band + RSI oversold/bought signals  
- ✅ **S6 (Breakout)** - Volume-confirmed range breakouts
- ✅ **S11 (Momentum)** - Trend continuation with acceleration

### **Symbol-Specific Learning:**
- ✅ **ES (E-mini S&P 500)** - High liquidity, tight spreads
- ✅ **NQ (E-mini Nasdaq-100)** - Higher volatility, tech sector exposure

### **What's Being Optimized:**
- 🎯 **Position sizing** - Risk-adjusted contract quantities per signal
- 📊 **Strategy selection** - Which strategy to use given market conditions  
- ⚡ **Execution quality** - Optimal entry/exit timing and sizing
- 🛡️ **Risk management** - Dynamic stop placement and position limits

---

## 🔧 **Configuration & Monitoring**

### **Environment Variables (Local)**
```bash
# Enable both symbols
TOPSTEPX_ENABLE_NQ=1
SYMBOLS=ES,NQ

# Cloud learning integration  
RL_ENABLED=1
MODEL_POLL_SEC=7200  # Check for new models every 2 hours

# Model paths
RL_ONNX=models/rl/latest_rl_sizer.onnx
```

### **GitHub Secrets (Cloud)**
```bash
AWS_ACCESS_KEY_ID      # S3 access
AWS_SECRET_ACCESS_KEY  # S3 access  
S3_BUCKET             # Model storage
MANIFEST_HMAC_KEY     # Security
```

### **Monitoring Endpoints**
- `/healthz` - Overall system health
- `/healthz/canary` - A/B testing status
- `/verify/today` - Trading performance summary
- `/build` - Version and mode information

---

## 🎯 **Success Metrics**

### **Learning Indicators:**
- ✅ New models deployed every 30 minutes (when data available)
- ✅ Symbol distribution shows both ES and NQ data
- ✅ Strategy performance tracked per symbol
- ✅ Position sizing adapts to market conditions
- ✅ Risk-adjusted returns improve over time

### **Performance Tracking:**
- 📊 **Win Rate**: % of profitable trades per strategy/symbol
- 📈 **R-Multiple**: Risk-adjusted returns (target: >1.0)
- 🎯 **Sharpe Ratio**: Risk-adjusted performance
- 📉 **Drawdown**: Maximum losing streak management
- 🔄 **Adaptation**: Model performance vs baseline

---

## 🚀 **Quick Verification Checklist**

To verify your bot is learning continuously on both symbols:

1. **Check GitHub Actions**: [Repository → Actions] - Should show runs every 30 minutes
2. **Verify Symbol Data**: Log files should show both ES and NQ features
3. **Monitor Model Updates**: Check `models/` directory for new ONNX files
4. **Trading Activity**: Both symbols should show active signal generation
5. **Performance Metrics**: `/verify/today` should show activity on both symbols

Your bot is now a **learning machine** that gets smarter every 30 minutes! 🧠⚡