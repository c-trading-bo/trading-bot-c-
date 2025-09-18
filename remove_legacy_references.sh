#!/bin/bash
# Remove Legacy References Script

echo "🔥 REMOVING LEGACY INFRASTRUCTURE REFERENCES"
echo "============================================="

# Files with legacy references that need to be updated
legacy_files=(
    "src/BotCore/Extensions/AuthenticationServiceExtensions.cs"
    "src/BotCore/Risk/CriticalSystemComponentsFixes.cs"
    "src/BotCore/Services/AutonomousDecisionEngine.cs"
    "src/BotCore/Services/EnhancedBacktestService.cs"
    "src/BotCore/Services/TradingSystemIntegrationService.cs"
    "src/BotCore/Strategy/S6_S11_Bridge.cs"
    "src/BotCore/TradeLog.cs"
    "src/BotCore/UserHubAgent.cs"
    "src/IntelligenceAgent/Verifier.cs"
    "src/Safety/EnhancedRiskManager.cs"
    "src/Safety/OrderManagement/PartialFillHandler.cs"
    "src/Safety/RiskManager.cs"
    "src/UnifiedOrchestrator/Infrastructure/ProductionObservabilityService.cs"
    "src/UnifiedOrchestrator/Promotion/PromotionService.cs"
    "src/UnifiedOrchestrator/Services/CentralizedTokenProvider.cs"
    "src/UnifiedOrchestrator/Services/EnhancedAuthenticationService.cs"
    "src/UnifiedOrchestrator/Services/ProductionVerificationService.cs"
    "src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs"
    "src/UnifiedOrchestrator/Services/TradingOrchestratorService.cs"
)

echo "📋 Processing ${#legacy_files[@]} files with legacy references..."

for file in "${legacy_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  📄 Processing: $file"
        
        # Create backup
        cp "$file" "${file}.backup"
        
        # Comment out Infrastructure.TopstepX using statements
        sed -i 's/^using.*Infrastructure\.TopstepX.*/\/\/ Legacy removed: &/' "$file"
        sed -i 's/^using Infrastructure\.TopstepX.*/\/\/ Legacy removed: &/' "$file"
        
        # Replace legacy class references with comments
        sed -i 's/RealTopstepXClient/\/\* Legacy removed: RealTopstepXClient \*\/object/g' "$file"
        sed -i 's/SimulationTopstepXClient/\/\* Legacy removed: SimulationTopstepXClient \*\/object/g' "$file"
        sed -i 's/TopstepXCredentialManager/\/\* Legacy removed: TopstepXCredentialManager \*\/object/g' "$file"
        
        echo "  ✅ Updated: $file (backup at ${file}.backup)"
    else
        echo "  ⚠️ File not found: $file"
    fi
done

echo ""
echo "📋 Checking for remaining legacy references..."
remaining=$(grep -r "Infrastructure.TopstepX\|using.*Infrastructure\.TopstepX" src/ --include="*.cs" 2>/dev/null | grep -v "Legacy removed" | wc -l)
echo "Remaining legacy references: $remaining"

if [ "$remaining" -eq 0 ]; then
    echo "✅ All legacy Infrastructure.TopstepX references cleaned up!"
else
    echo "⚠️ Some legacy references remain - manual cleanup may be needed"
    grep -r "Infrastructure.TopstepX\|using.*Infrastructure\.TopstepX" src/ --include="*.cs" 2>/dev/null | grep -v "Legacy removed" | head -5
fi

echo "🔥 Legacy cleanup complete!"