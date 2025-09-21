#!/bin/bash

# Test CA2007 ConfigureAwait fix on a single file
# This is a minimal test before running the full checkpoint

set -e

echo "🧪 Testing CA2007 ConfigureAwait fix..."

# Target specific violations
FILE1="src/IntelligenceStack/LeaderElectionService.cs"
FILE2="src/IntelligenceStack/ModelQuarantineManager.cs"

if [ ! -f "$FILE1" ]; then
    echo "❌ File not found: $FILE1"
    exit 1
fi

echo "📊 Current CA2007 violations:"
dotnet build src/IntelligenceStack/IntelligenceStack.csproj 2>&1 | grep "CA2007" | wc -l

echo "🔧 Applying ConfigureAwait fix to $FILE1..."

# Backup
cp "$FILE1" "$FILE1.backup"

# Apply specific fixes to line 488 area
sed -i 's/return await Task\.Run(/return await Task.Run(/g' "$FILE1"
sed -i 's/await Task\.Run([^)]*)/&.ConfigureAwait(false)/g' "$FILE1"

echo "✅ Fix applied, checking build..."

if dotnet build src/IntelligenceStack/IntelligenceStack.csproj --verbosity quiet > /dev/null 2>&1; then
    echo "✅ Build successful"
    
    echo "📊 New CA2007 violations:"
    NEW_COUNT=$(dotnet build src/IntelligenceStack/IntelligenceStack.csproj 2>&1 | grep "CA2007" | wc -l)
    echo "$NEW_COUNT violations remaining"
    
    echo "💾 Keeping changes"
    rm "$FILE1.backup"
else
    echo "❌ Build failed, reverting changes"
    mv "$FILE1.backup" "$FILE1"
    exit 1
fi