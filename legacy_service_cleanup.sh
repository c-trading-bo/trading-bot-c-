#!/bin/bash
# Legacy Service Cleanup - Remove or Disable Legacy Services

echo "🔥 LEGACY SERVICE CLEANUP - Removing/Disabling Legacy Services"
echo "=============================================================="

echo "📋 Phase 1: Disable legacy services in Program.cs registration..."

# Create backup
cp src/UnifiedOrchestrator/Program.cs src/UnifiedOrchestrator/Program.cs.backup

echo "📋 Phase 2: Comment out or remove legacy service files..."

# List of legacy service files to handle
legacy_services=(
    "src/UnifiedOrchestrator/Services/EnhancedAuthenticationService.cs"
    "src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs"
    "src/UnifiedOrchestrator/Services/CentralizedTokenProvider.cs"
    "src/UnifiedOrchestrator/Services/TradingOrchestratorService.cs"
    "src/UnifiedOrchestrator/Services/ProductionVerificationService.cs"
    "src/UnifiedOrchestrator/Infrastructure/ProductionObservabilityService.cs"
    "src/UnifiedOrchestrator/Promotion/PromotionService.cs"
)

for service in "${legacy_services[@]}"; do
    if [ -f "$service" ]; then
        echo "  📄 Found legacy service: $service"
        # Move to backup instead of deleting for safety
        mv "$service" "${service}.legacy_backup"
        echo "  ✅ Moved to backup: ${service}.legacy_backup"
    fi
done

echo "📋 Phase 3: Check remaining legacy references..."
echo "Remaining Infrastructure.TopstepX references:"
grep -r "Infrastructure.TopstepX\|using.*Infrastructure\.TopstepX" src/ --include="*.cs" 2>/dev/null | cut -d: -f1 | sort | uniq || echo "None found"

echo ""
echo "🔥 Legacy service cleanup complete!"