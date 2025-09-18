#!/bin/bash
# Complete Legacy Code Purge Script

echo "🔥 COMPLETE LEGACY CODE PURGE - Infrastructure.TopstepX Removal"
echo "=================================================================="

# Step 1: Remove the entire Infrastructure.TopstepX project
echo "📋 Step 1: Removing Infrastructure.TopstepX project completely..."
if [ -d "src/Infrastructure.TopstepX" ]; then
    rm -rf src/Infrastructure.TopstepX
    echo "✅ Infrastructure.TopstepX project removed"
else
    echo "ℹ️ Infrastructure.TopstepX project already removed"
fi

# Step 2: Remove from solution file
echo "📋 Step 2: Removing from solution file..."
if grep -q "Infrastructure.TopstepX" TopstepX.Bot.sln; then
    sed -i '/Infrastructure\.TopstepX/d' TopstepX.Bot.sln
    echo "✅ Removed Infrastructure.TopstepX from solution"
else
    echo "ℹ️ Infrastructure.TopstepX already removed from solution"
fi

# Step 3: Remove project references from all .csproj files
echo "📋 Step 3: Removing project references from .csproj files..."
find . -name "*.csproj" -exec sed -i '/Infrastructure\.TopstepX/d' {} \;
echo "✅ Removed Infrastructure.TopstepX references from .csproj files"

# Step 4: List files that still reference Infrastructure.TopstepX or legacy classes
echo "📋 Step 4: Finding remaining legacy references..."
echo "Files still referencing Infrastructure.TopstepX:"
grep -r "Infrastructure.TopstepX\|using.*Infrastructure\.TopstepX" src/ --include="*.cs" 2>/dev/null | cut -d: -f1 | sort | uniq || echo "None found"

echo ""
echo "Files referencing legacy TopstepX classes:"
grep -r "RealTopstepXClient\|SimulationTopstepXClient\|TopstepXCredentialManager" src/ --include="*.cs" 2>/dev/null | cut -d: -f1 | sort | uniq || echo "None found"

echo ""
echo "🔥 Legacy purge analysis complete!"