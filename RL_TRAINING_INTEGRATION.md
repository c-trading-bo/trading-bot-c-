## 🎯 **Training Data Integration Guide**

### **How Your C# Bot Collects Training Data**

Your bot already has signal generation and trade logging. I've added `RlTrainingDataCollector` to capture the additional data needed for RL training.

### **Step 1: Integrate Data Collection (C#)**

**In your signal generation code** (around Program.cs line 2752), add this after `TradeLog.Signal`:

```csharp
try { 
    BotCore.TradeLog.Signal(log, symbol, chosen.StrategyId, chosen.Side, chosen.Size, chosen.Entry, chosen.Stop, chosen.Target, $"score={chosen.ExpR:F2}", chosen.Tag ?? string.Empty); 
    
    // ✅ ADD: Collect RL training features
    var signalId = $"{symbol}_{chosen.StrategyId}_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
    var features = BotCore.RlTrainingDataCollector.CreateFeatureSnapshot(
        signalId, symbol, chosen.StrategyId, chosen.Entry, baselineMultiplier: 1.0m);
    
    // TODO: Populate with your actual market data
    features.Atr = GetAtr(symbol);           // From your indicators
    features.Rsi = GetRsi(symbol);           // From your indicators  
    features.SignalStrength = chosen.ExpR;   // Your signal score
    features.PriorWinRate = GetWinRate(chosen.StrategyId); // Your stats
    
    BotCore.RlTrainingDataCollector.LogFeatures(log, features);
} catch { }
```

**In your trade outcome handling** (when positions close), add:

```csharp
// When a trade completes
var outcome = new BotCore.RlTrainingDataCollector.TradeOutcome
{
    SignalId = signalId,  // Match the signal
    EntryTime = entryTime,
    ExitTime = DateTime.UtcNow,
    EntryPrice = entryPrice,
    ExitPrice = exitPrice,
    StopPrice = stopPrice,
    TargetPrice = targetPrice,
    RMultiple = CalculateRMultiple(entryPrice, exitPrice, stopPrice, side),
    SlippageTicks = CalculateSlippage(entryPrice, exitPrice),
    IsWin = exitPrice > entryPrice, // Adjust for side
    IsCompleted = true,
    ExitReason = "Target" // or "Stop", "Manual", etc.
};

BotCore.RlTrainingDataCollector.LogOutcome(log, outcome);
```

### **Step 2: Data Files Generated**

Your bot will create training data files:
```
data/rl_training/
├── features_20250830.jsonl    # Daily feature snapshots
├── outcomes_20250830.jsonl    # Daily trade outcomes  
└── training_data_20250801_20250830.csv  # Merged training set
```

### **Step 3: Export Training Data**

**Monthly training data export** (add to your bot's maintenance routine):

```csharp
// Export last 30 days of data for training
var csvPath = BotCore.RlTrainingDataCollector.ExportToCsv(log, 
    DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);
log.LogInformation("RL training data exported: {Path}", csvPath);
```

### **Step 4: Python Training (Command Line)**

**Once you have 2-4 weeks of data:**

```bash
# Train RL model on your historical data
cd "C:\Users\kevin\Downloads\C# ai bot\ml"
.\rl_env\Scripts\python.exe rl\train_cvar_ppo.py \
    --data "data\rl_training\training_data_20250801_20250830.csv" \
    --output_model "models\rl\latest_rl_sizer.onnx"
```

### **Step 5: Model Deployment (Automatic)**

Your `RlSizer` class automatically detects new models:
- ✅ Hot-reloads when `models/rl/latest_rl_sizer.onnx` is updated
- ✅ Falls back to baseline sizing if model missing
- ✅ Applies safety bounds (0.1x-2.0x multiplier)

### **Complete Workflow:**

```
Day 1-30:  🤖 C# Bot collects features + outcomes
Day 30:    📊 Export training data to CSV  
Day 30:    🧠 Python trains RL model → .onnx file
Day 31+:   🚀 C# Bot uses new RL model for position sizing
```

**Your C# bot does the heavy lifting - Python just processes the data once in a while!**
