#!/bin/bash
# Legacy Code Cleanup Script - Remove Infrastructure.TopstepX

echo "🔥 LEGACY CODE CLEANUP - Removing Infrastructure.TopstepX"
echo "=================================================="

# Phase 1: Remove project from solution
echo "📋 Phase 1: Removing Infrastructure.TopstepX from solution..."
sed -i '/Infrastructure\.TopstepX/d' TopstepX.Bot.sln

# Phase 2: Remove project references from .csproj files
echo "📋 Phase 2: Removing project references..."
find . -name "*.csproj" -exec sed -i '/Infrastructure\.TopstepX/d' {} \;

# Phase 3: List files that still reference Infrastructure.TopstepX
echo "📋 Phase 3: Finding remaining references..."
echo "Files still referencing Infrastructure.TopstepX:"
grep -r "Infrastructure.TopstepX\|using.*TopstepX" src/ --include="*.cs" | cut -d: -f1 | sort | uniq

echo "📋 Phase 4: Files referencing legacy TopstepX classes:"
grep -r "RealTopstepXClient\|SimulationTopstepXClient\|TopstepXService\|TopstepXCredentialManager" src/ --include="*.cs" | cut -d: -f1 | sort | uniq

echo "🔥 Legacy cleanup analysis complete!"