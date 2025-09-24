#!/bin/bash

echo "================================================================================"
echo "🚀 UNIFIED ORCHESTRATOR - PRODUCTION LAUNCH SEQUENCE"
echo "================================================================================"
echo

# Set production environment
export DOTNET_ENVIRONMENT=Production
export ASPNETCORE_ENVIRONMENT=Production

# Ensure kill switch is inactive for intentional launch
if [ -f "kill.txt" ]; then
    echo "⚠️  Kill switch detected (kill.txt exists). Remove to proceed with launch."
    echo "   Run: rm kill.txt"
    exit 1
fi

echo "✅ Environment: Production"
echo "✅ Kill Switch: Inactive"
echo

# Build with production optimizations
echo "🔨 Building UnifiedOrchestrator with production optimizations..."
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj \
    --configuration Release \
    --verbosity minimal \
    -p:TreatWarningsAsErrors=false \
    -p:CodeAnalysisTreatWarningsAsErrors=false

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Check compilation errors above."
    exit 1
fi

echo "✅ Build completed successfully"
echo

# Launch the UnifiedOrchestrator
echo "🎯 Launching UnifiedOrchestrator in Production Mode..."
echo "================================================================================"

dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj \
    --configuration Release \
    -p:TreatWarningsAsErrors=false \
    -p:CodeAnalysisTreatWarningsAsErrors=false \
    "$@"