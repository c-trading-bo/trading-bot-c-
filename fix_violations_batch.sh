#!/bin/bash
# Batch fix script for common analyzer violations
# Focus on simple pattern-based fixes that can be automated

set -e

echo "🔧 Starting batch violation fixes..."

# Fix CA1840: Thread.CurrentThread.ManagedThreadId → Environment.CurrentManagedThreadId
echo "Fixing CA1840 violations..."
find src/IntelligenceStack -name "*.cs" -exec sed -i 's/Thread\.CurrentThread\.ManagedThreadId/Environment.CurrentManagedThreadId/g' {} \;

# Fix simple S1854: Remove useless assignments to _
echo "Fixing S1854 violations (discard pattern)..."
find src/IntelligenceStack -name "*.cs" -exec sed -i 's/var _ = /_ = /g' {} \;

# Fix CA1024: Methods that should be properties (simple cases)
echo "Fixing simple CA1024 violations..."
# This would need more complex pattern matching, skip for now

echo "✅ Batch fixes completed!"

# Run a build to check results
echo "📊 Checking violation counts after batch fixes..."
dotnet build src/IntelligenceStack/IntelligenceStack.csproj 2>&1 | grep -E "error (CA|S|AsyncFixer)[0-9]+" | sed -E 's/.*error ([A-Za-z0-9]+):.*/\1/' | sort | uniq -c | sort -nr > batch_fix_results.txt
echo "Results saved to batch_fix_results.txt"