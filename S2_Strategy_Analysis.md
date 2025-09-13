# 📊 S2 Strategy - VWAP Mean Reversion Strategy

## 🎯 **Strategy Overview**
Your **S2 strategy** is a sophisticated **VWAP (Volume Weighted Average Price) mean reversion** strategy that fades extreme moves away from VWAP and bets on price returning to the average.

## ⚙️ **Core Strategy Logic**

### **Entry Conditions:**
- **LONG (Buy)**: When price is stretched **below VWAP** by 2+ standard deviations
- **SHORT (Sell)**: When price is stretched **above VWAP** by 2+ standard deviations

### **Key Parameters:**
```csharp
// Core Mean Reversion Settings
SigmaEnter = 2.0m;           // Minimum σ distance from VWAP to enter
AtrEnter = 1.0m;             // Alternative ATR-based entry threshold
SigmaForceTrend = 2.8m;      // Higher threshold during trend days

// Instrument-Specific Thresholds
EsSigma = 2.0m;              // ES-specific sigma requirement
NqSigma = 2.6m;              // NQ needs higher threshold (more volatile)

// Risk Management
StopAtrMult = 0.75m;         // Stop loss at 0.75x ATR
TrailAtrMult = 1.0m;         // Trailing stop at 1.0x ATR
MaxBarsInTrade = 45;         // Maximum 45 bars in position
```

## 🔧 **Advanced Features**

### **1. Session-Based VWAP Calculation**
- Anchors VWAP to **9:30 AM ET** market open
- Calculates volume-weighted standard deviation (σ)
- Uses **session VWAP**, not rolling VWAP

### **2. Trend Day Protection**
```csharp
// Detects trend days using EMA20 slope
decimal slope = (ema20[^1] - ema20[^6]) / 5m;
bool strongTrend = Math.Abs(slopeTicks) > MinSlopeTf2;

// Raises sigma requirement on trend days
if (strongTrend) needSigma = Math.Max(needSigma, SigmaForceTrend);
```

### **3. Volume Regime Filtering**
```csharp
// Only trades in normal volume conditions
VolZMin = -0.3m;             // Minimum volume Z-score
VolZMax = 2.2m;              // Maximum volume Z-score
```

### **4. Microstructure Safeguards**
- **Volume Imbalance**: Checks up/down volume ratio
- **Pivot Distance**: Requires distance from recent swing points
- **Prior Day Extremes**: Avoids areas near yesterday's high/low
- **Spread Control**: Rejects if bid-ask spread too wide

### **5. Time-Based Filters**
```csharp
// Initial Balance continuation after 10:30 AM
if (nowMin >= IbEndMinute) {
    // Avoids fading strong IB breakouts
    // Only allows extreme fades (>2.8σ)
}

// Curfew system for overnight risk
CurfewNoNewHHMM = "09:15";    // No new trades after 9:15 AM
```

## 📈 **Entry Logic Flow**

### **LONG Setup (Fade Down Move):**
1. ✅ Price below VWAP by 2+ sigma OR 1+ ATR
2. ✅ Volume imbalance favors buyers (≥0.9 ratio)
3. ✅ Sufficient distance from swing lows
4. ✅ Room vs prior day extremes
5. ✅ Z-score deceleration (momentum slowing)
6. ✅ Bull confirmation OR reclaim of -2σ level

### **SHORT Setup (Fade Up Move):**
1. ✅ Price above VWAP by 2+ sigma OR 1+ ATR
2. ✅ Volume imbalance favors sellers (≤1.1 ratio)
3. ✅ Sufficient distance from swing highs
4. ✅ Room vs prior day extremes
5. ✅ Z-score deceleration (momentum slowing)
6. ✅ Bear confirmation OR rejection at +2σ level

## 🎲 **Risk Management**

### **Position Sizing:**
- Uses **Average Daily Range (ADR)** for context
- Requires minimum 25% ADR room to target
- Blocks if today's range > 120% of ADR (exhaustion)

### **Stop Losses:**
```csharp
// LONG stops
var swing = bars.Min(b => b.Low);        // Recent swing low
var dn3 = vwap - 3m * sigma;            // 3-sigma level
var stop = Math.Min(swing, dn3);        // Tighter of the two

// SHORT stops  
var swing = bars.Max(b => b.High);       // Recent swing high
var up3 = vwap + 3m * sigma;            // 3-sigma level
var stop = Math.Max(swing, up3);        // Tighter of the two
```

### **Profit Targets:**
- **Primary Target**: Return to **VWAP**
- **Minimum R:R**: 0.8:1 (adjusts if too tight)

## 🔍 **Smart Adaptations**

### **Roll Week Adjustments:**
- Detects futures expiration weeks
- Increases sigma requirements by 0.3 during roll periods

### **Peer Correlation:**
- Monitors ES/NQ correlation
- Blocks fades against strong peer moves
- Prevents fighting coordinated index moves

### **Dynamic Thresholds:**
```csharp
// Adjusts based on market conditions
decimal dynSigma = DynamicSigmaThreshold(
    baseSigma,    // Base requirement
    volz,         // Volume regime
    slope5,       // Trend strength  
    timeOfDay,    // Session effects
    symbol        // Instrument type
);
```

## 📊 **Strategy Strengths**

### ✅ **Advantages:**
- **High Win Rate**: Mean reversion has natural edge
- **Volume Confirmation**: Uses volume for trade quality
- **Trend Awareness**: Avoids fading strong trends
- **Risk Controlled**: Multiple stop mechanisms
- **Session Context**: Proper VWAP calculation

### ⚠️ **Considerations:**
- **Trend Day Risk**: Can struggle in strong trending markets
- **Overnight Gaps**: Vulnerable to gap moves
- **Low Frequency**: Selective entry criteria = fewer trades
- **Parameter Sensitivity**: Needs tuning for different market regimes

## 🎯 **Optimization Areas**
1. **Entry Timing**: "Retest mode" waits for micro pullbacks
2. **Exit Optimization**: Could add partial profit taking
3. **Regime Detection**: Could adapt faster to changing conditions
4. **Multi-Timeframe**: Could use higher TF for context

Your S2 strategy is a **sophisticated, institutional-quality** mean reversion system with excellent risk controls and market awareness! 🚀