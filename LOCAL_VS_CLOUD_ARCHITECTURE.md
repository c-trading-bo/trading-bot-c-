# 🏠 Local vs ☁️ Cloud Architecture Guide

## 📋 Overview

Your trading bot uses **100% cloud-based learning** combined with local execution for optimal performance. You only need to run the bot when actively trading - all learning happens automatically in the cloud.

## 🏠 **What Stays LOCAL (Real-Time Execution Only)**

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

### 📊 **Data Collection & Upload**
- **Feature snapshots** - Market data at signal generation time
- **Trade outcomes** - Performance tracking for feedback loops
- **Symbol-specific data** - Both ES and NQ training data collection
- **Automatic cloud upload** - Training data uploaded to S3 every 15 minutes

**Why Local?** 
- ⚡ **Speed**: Sub-millisecond decision making
- 🔒 **Reliability**: No network dependencies for critical operations
- 💰 **Cost**: No compute charges for real-time operations
- 🎯 **Focus**: Bot only runs when actively trading, not 24/7 for learning

---

## ☁️ **What Runs in CLOUD (100% Learning Pipeline)**

### 🤖 **Model Training Pipeline**
- **Feature engineering** - Advanced technical indicators and market regime detection
- **Deep learning training** - GPU-accelerated neural networks
- **Reinforcement learning** - Position sizing optimization using PPO/A2C
- **Cross-validation** - Robust model validation across time periods
- **Hyperparameter optimization** - Automated tuning for best performance

### 🔄 **Continuous Learning (Every 30 Minutes, 24/7)**
- **Automatic data ingestion** - Pulls training data uploaded from all bot instances
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
- 📈 **Scale**: Handle massive datasets from multiple bot instances
- 🔄 **Availability**: 24/7 training even when local bot is offline
- 💡 **Intelligence**: Advanced ML algorithms that require significant compute
- 🎯 **Efficiency**: No need to run bot 24/7 just for learning

---

## 🔄 **How They Work Together**

### **Data Flow: Local → Cloud**
1. **Local bot** generates trading signals and collects features during active trading
2. **Training data** logged to symbol-specific files (`features_es_*.jsonl`, `features_nq_*.jsonl`)
3. **Trade outcomes** recorded for performance feedback
4. **CloudDataUploader** converts JSONL to parquet and uploads to S3 every 15 minutes
5. **GitHub Actions** processes uploaded data in cloud training pipeline every 30 minutes

### **Model Flow: Cloud → Local**
1. **Cloud training** produces optimized ONNX models every 30 minutes
2. **Models uploaded** to S3 with cryptographic signatures
3. **CloudRlTrainerEnhanced** automatically downloads new models via manifest checking
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

# Cloud learning integration (download models, upload data)
RL_ENABLED=1
MODEL_POLL_SEC=7200  # Check for new models every 2 hours

# S3 configuration for data upload
S3_BUCKET=your-training-data-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

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
- ✅ Training data automatically uploaded every 15 minutes
- ✅ Symbol distribution shows both ES and NQ data
- ✅ Strategy performance tracked per symbol
- ✅ Position sizing adapts to market conditions
- ✅ Risk-adjusted returns improve over time
- ✅ No local training required - 100% cloud-based

### **Performance Tracking:**
- 📊 **Win Rate**: % of profitable trades per strategy/symbol
- 📈 **R-Multiple**: Risk-adjusted returns (target: >1.0)
- 🎯 **Sharpe Ratio**: Risk-adjusted performance
- 📉 **Drawdown**: Maximum losing streak management
- 🔄 **Adaptation**: Model performance vs baseline

---

## 🚀 **Quick Verification Checklist**

To verify your bot is learning continuously with 100% cloud-based training:

1. **Check GitHub Actions**: [Repository → Actions] - Should show training runs every 30 minutes
2. **Verify Data Upload**: Check S3 bucket for uploaded training data (parquet files)
3. **Monitor Model Updates**: Check `models/` directory for new ONNX files
4. **Trading Activity**: Both symbols should show active signal generation
5. **Performance Metrics**: `/verify/today` should show activity on both symbols
6. **No Local Training**: AutoRlTrainer should show "DEPRECATED" warnings if enabled

**🎯 Key Benefit**: Your bot only needs to run during trading hours. All learning happens automatically in the cloud! 🌥️⚡