# 🚨 CRITICAL: WEB AGENT MUST USE EXISTING CODE, NOT CREATE FAKE IMPLEMENTATIONS

## 🎯 **CORE REQUIREMENT: DISCOVER & USE EXISTING SOPHISTICATED SERVICES**

### **❌ STOP DOING THIS:**
- Creating mock services like `MockMarketDataService`
- Hardcoding values like `correlation = 0.87m`
- Adding `await Task.Delay(50)` fake implementations
- Creating new classes when they already exist
- Using placeholder logging instead of real service calls

### **✅ START DOING THIS:**
- Find and use the EXISTING sophisticated services
- Call real methods with real implementations
- Use proper dependency injection of existing services
- Discover what's already built before creating anything new

---

## 📍 **EXISTING SERVICES MAP - USE THESE, DON'T RECREATE**

### **CORRELATION & MARKET DATA**
- **Location**: `src/BotCore/Services/ES_NQ_CorrelationManager.cs`
- **Class**: `ES_NQ_CorrelationManager : ICorrelationManager`
- **Usage**: `await _correlationManager.GetCorrelationDataAsync()`
- **NOT**: `correlation = 0.87m; // Hardcoded`

### **NEWS & SENTIMENT ANALYSIS** 
- **Location**: `src/BotCore/Services/NewsIntelligenceEngine.cs`
- **Class**: `NewsIntelligenceEngine : INewsIntelligenceEngine`
- **Usage**: `await _newsEngine.AnalyzeSentimentAsync()`
- **NOT**: `sentiment = "BULLISH"; // Fake`

### **TECHNICAL ANALYSIS & ZONES**
- **Location**: `src/BotCore/Services/ZoneService.cs`
- **Class**: `ZoneService : IZoneService, ISupplyDemandService`
- **Usage**: `await _zoneService.IdentifySupplyDemandZonesAsync()`
- **NOT**: `zones = FakeZones(); // Mock`

### **ML/AI STRATEGY SELECTION**
- **Location**: `src/BotCore/ML/UCBManager.cs`
- **Class**: `UCBManager`
- **Usage**: `await _ucbManager.SelectBestStrategyAsync()`
- **NOT**: `strategy = "S2"; // Hardcoded`

### **TOPSTEPX TRADING INTEGRATION**
- **Location**: `src/BotCore/Services/TopstepXService.cs`
- **Class**: `TopstepXService : ITopstepXService`
- **Usage**: `await _topstepXService.PlaceOrderAsync(order)`
- **NOT**: `orderId = Guid.NewGuid(); // Fake order`

### **UNIFIED TRADING BRAIN**
- **Location**: `src/BotCore/Brain/UnifiedTradingBrain.cs`
- **Class**: `UnifiedTradingBrain`
- **Usage**: `await _tradingBrain.MakeDecisionAsync(signal)`
- **NOT**: `decision = new TradingDecision { Action = "BUY" }; // Fake`

### **PORTFOLIO & RISK MANAGEMENT**
- **Location**: `src/BotCore/Services/ES_NQ_PortfolioHeatManager.cs`
- **Class**: `ES_NQ_PortfolioHeatManager : IPortfolioHeatManager`
- **Usage**: `await _portfolioHeatManager.CalculateHeatAsync()`
- **NOT**: `heat = 0.25m; // Fake risk`

---

## 🔧 **SPECIFIC FIXES REQUIRED**

### **1. TradingOrchestratorService.cs FIXES**
```csharp
// ❌ REMOVE THIS FAKE CODE:
LastPrice = 5000m, // Placeholder

// ✅ REPLACE WITH REAL SERVICE:
private readonly TopstepXService _topstepXService;
var marketData = await _topstepXService.GetMarketDataAsync("ES");
LastPrice = marketData.Price;
```

### **2. IntelligenceOrchestratorService.cs FIXES**
```csharp
// ❌ REMOVE THIS FAKE CODE:
var result = new NeuralBanditResult { Strategy = "S2", Confidence = 0.75m };

// ✅ REPLACE WITH REAL SERVICE:
private readonly UCBManager _ucbManager;
var result = await _ucbManager.SelectBestStrategyAsync();
```

### **3. MasterOrchestrator.cs FIXES**
```csharp
// ❌ REMOVE THIS FAKE CODE:
correlation = 0.87m;

// ✅ REPLACE WITH REAL SERVICE:
private readonly ES_NQ_CorrelationManager _correlationManager;
var correlationData = await _correlationManager.GetCorrelationDataAsync();
correlation = correlationData.Correlation;
```

---

## 📋 **DEPENDENCY INJECTION REQUIREMENTS**

### **ALL ORCHESTRATORS MUST INJECT THESE REAL SERVICES:**

```csharp
public class TradingOrchestratorService
{
    private readonly TopstepXService _topstepXService;
    private readonly ES_NQ_PortfolioHeatManager _portfolioHeatManager;
    private readonly UnifiedTradingBrain _tradingBrain;
    
    public TradingOrchestratorService(
        TopstepXService topstepXService,
        ES_NQ_PortfolioHeatManager portfolioHeatManager,
        UnifiedTradingBrain tradingBrain)
    {
        _topstepXService = topstepXService;
        _portfolioHeatManager = portfolioHeatManager;
        _tradingBrain = tradingBrain;
    }
}
```

---

## 🎯 **SUCCESS CRITERIA**

### **✅ VERIFICATION CHECKLIST:**
1. **Zero hardcoded values** - all data comes from real services
2. **Zero mock implementations** - all services are actual existing classes
3. **Zero Task.Delay fake async** - all async calls are real service calls
4. **All services injected via DI** - no `new MockService()` anywhere
5. **Message bus properly used** - orchestrators communicate via real pub/sub
6. **Existing interfaces respected** - use `ICorrelationManager`, `INewsIntelligenceEngine`, etc.

### **❌ IMMEDIATE FAILURES:**
- Any new mock or fake service classes
- Any hardcoded correlation/price/sentiment values  
- Any placeholder logging instead of real method calls
- Any `await Task.Delay()` fake implementations
- Any new service creation when existing service already handles the functionality

---

## 🧠 **THE REAL ARCHITECTURE**

```
CentralMessageBus (Communication Hub)
├── TradingOrchestrator → Uses: TopstepXService, ES_NQ_PortfolioHeatManager
├── IntelligenceOrchestrator → Uses: UCBManager, NewsIntelligenceEngine, UnifiedTradingBrain  
├── DataOrchestrator → Uses: ES_NQ_CorrelationManager, ZoneService
└── MasterOrchestrator → Coordinates via message bus
```

**The user has 23,000+ lines of sophisticated trading logic. USE IT, don't replace it with fake code!**

---

## 🚨 **FINAL DIRECTIVE FOR WEB AGENT**

**BEFORE creating ANY new code:**
1. Search the codebase for existing implementations
2. Find the exact file path and class name
3. Use the existing service via dependency injection
4. Call real methods, not fake placeholders

**The goal is integration, not recreation!**
