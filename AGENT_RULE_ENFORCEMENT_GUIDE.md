# Agent Rule Enforcement System - Usage Guide

## Overview

This document explains how the analyzer enforcement system prevents agents from disabling aggressive build rules and ensures configuration-driven ML/AI values instead of hardcoded ones.

## Problem Solved

**Issue #3304685224**: Agents were:
1. Causing duplicate sessions leading to double premium costs
2. Disabling aggressive build rules to force green builds  
3. Using hardcoded values (2.5, 0.7, 1.0) in AI/ML core systems

## Solution Components

### 1. Duplicate Agent Session Prevention ✅

**File**: `src/UnifiedOrchestrator/Services/UnifiedOrchestratorService.cs`

```csharp
// ✅ Correct Usage - prevents duplicate sessions
bool launched = orchestrator.TryLaunchAgent("agent-key", async () => {
    // Agent implementation
});

if (!launched) {
    // Duplicate prevented - no double premium cost
}
```

**Features**:
- Thread-safe session registry
- Duplicate detection and prevention
- Automatic cleanup on completion
- Audit logging for compliance

### 2. Analyzer Rule Enforcement 

**File**: `src/Safety/Analyzers/ProductionRuleEnforcementAnalyzer.cs`

**Prevents**:
```csharp
// ❌ BLOCKED: Unauthorized suppression
[SuppressMessage("Category", "Rule")] // Missing RuntimeProof

// ❌ BLOCKED: Pragma disable
#pragma warning disable CS1234

// ❌ BLOCKED: Commented rules in .editorconfig
# dotnet_diagnostic.S109.severity = error
```

**Allows**:
```csharp
// ✅ ALLOWED: With justification
[SuppressMessage("Category", "Rule", Justification = "RuntimeProof: Trading requires...")]
```

### 3. Configuration-Driven ML Values

**File**: `src/BotCore/Services/MLConfigurationService.cs`

**Old (Hardcoded)**:
```csharp
// ❌ BLOCKED by analyzer
return confidence >= 0.7;  // Hardcoded confidence
var positionSize = 2.5;    // Hardcoded position size
var regime = 1.0;          // Hardcoded regime detection
```

**New (Configuration)**:
```csharp
// ✅ CORRECT approach
return _mlConfig.IsConfidenceAcceptable(confidence);
var positionSize = _mlConfig.CalculatePositionSize(volatility, confidence, risk);
var isReliable = _mlConfig.IsRegimeDetectionReliable(regimeConfidence);
```

## Configuration Setup

### 1. Add to appsettings.json

```json
{
  "Trading": {
    "AIConfidenceThreshold": 0.75,
    "DefaultPositionSizeMultiplier": 2.0,
    "RegimeDetectionThreshold": 0.8,
    "StopLossBufferPercentage": 0.04,
    "RewardRiskRatioThreshold": 1.5
  }
}
```

### 2. Register in DI Container

```csharp
services.AddProductionConfigurationValidation(configuration);
services.AddScoped<MLConfigurationService>();
```

### 3. Use in Services

```csharp
public class MyTradingService
{
    private readonly MLConfigurationService _mlConfig;
    
    public MyTradingService(MLConfigurationService mlConfig)
    {
        _mlConfig = mlConfig;
    }
    
    public bool ShouldTrade(double confidence)
    {
        return _mlConfig.IsConfidenceAcceptable(confidence);
    }
}
```

## Build Rules Enforcement

### Current Rules

**Phase 1**: Hardcoded Business Values
- Detects: `PositionSize.*2.5`, `ModelConfidence.*0.7`, `RegimeDetection.*1.0`
- Action: Build failure with specific guidance

**Phase 2**: Rule Suppression Detection  
- Detects: `#pragma warning disable`, unauthorized `SuppressMessage`
- Action: Build failure requiring RuntimeProof justification

**Phase 3**: Configuration Downgrades
- Detects: Commented analyzer rules in `.editorconfig`
- Action: Build failure preventing rule circumvention

### Bypass Prevention

The analyzer cannot be bypassed by:
- Commenting out rules in `.editorconfig`
- Using `#pragma warning disable`
- Adding `SuppressMessage` without proper justification
- Downgrading severity to `none` or `silent`

## Testing & Verification

### Duplicate Prevention Test

```csharp
var result = await DuplicateSessionPreventionTest.VerifyDuplicatePreventionAsync(logger);
// Verifies only one session runs, others are blocked
```

### Configuration Test

```csharp
var demo = new MLConfigurationDemonstration(mlConfig);
var shouldTrade = demo.ShouldExecuteTrade(0.8); // Uses config, not hardcoded 0.7
```

## Compliance Monitoring

### Runtime Proof

1. **Session Logs**: Agent registry logs show duplicate prevention
2. **Build Logs**: Analyzer violations logged with specific guidance  
3. **Configuration Audit**: All ML values sourced from configuration
4. **Suppression Tracking**: All rule suppressions require justification

### Alerts

- Duplicate session attempts logged as warnings
- Analyzer violations fail build with specific error messages
- Configuration changes tracked in audit logs

## Troubleshooting

### Build Fails with "PRODUCTION VIOLATION"

```bash
PRODUCTION VIOLATION: Hardcoded business values detected. Use MLConfigurationService instead.
```

**Solution**: Replace hardcoded values with configuration service calls.

### Build Fails with "SuppressMessage without RuntimeProof"

```bash
SuppressMessage attribute found without RuntimeProof justification
```

**Solution**: Add proper justification or remove suppression.

### Duplicate Session Warning

```bash
🚫 [AGENT-REGISTRY] Duplicate launch prevented for agentKey: agent-1
```

**Resolution**: This is working correctly - preventing double premium costs.

## Best Practices

1. **Always use MLConfigurationService** for AI/ML parameters
2. **Include RuntimeProof justification** for any necessary suppressions  
3. **Test duplicate prevention** in your agent launch code
4. **Monitor build logs** for analyzer enforcement
5. **Keep configuration externalized** - never hardcode business values

## Files Modified

- `Directory.Build.props` - Build rule enforcement
- `src/Safety/Analyzers/ProductionRuleEnforcementAnalyzer.cs` - Custom analyzer
- `src/BotCore/Configuration/ProductionConfigurationValidation.cs` - ML config parameters
- `src/BotCore/Services/MLConfigurationService.cs` - Configuration service
- `src/BotCore/Services/IntelligenceService.cs` - Example usage
- `appsettings.production-ml.json` - Configuration example

## Success Metrics

✅ **Zero duplicate sessions** - No double premium costs  
✅ **Build rule enforcement** - No unauthorized suppressions  
✅ **Configuration-driven ML** - No hardcoded business values  
✅ **Runtime proof** - Full audit trail of all operations