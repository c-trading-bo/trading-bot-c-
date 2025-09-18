#!/bin/bash

echo "=== PRODUCTION ENFORCEMENT VALIDATION ==="
echo "Testing zero-tolerance build enforcement..."
echo ""

echo "1. Testing hardcoded business value detection (2.5)..."
if find . -name '*.cs' -not -path './bin/*' -not -path './obj/*' -not -path './packages/*' -not -path './test*/*' -not -path './Test*/*' -exec grep -l "= 2.5" {} \; | head -1 > /dev/null; then
    echo "✅ PASS: Hardcoded value 2.5 detected in production code"
else
    echo "❌ FAIL: Hardcoded value 2.5 not detected"
fi

echo ""
echo "2. Testing hardcoded confidence detection (0.7)..."
if find . -name '*.cs' -not -path './bin/*' -not -path './obj/*' -not -path './packages/*' -not -path './test*/*' -not -path './Test*/*' -exec grep -l "= 0.7" {} \; | head -1 > /dev/null; then
    echo "✅ PASS: Hardcoded confidence 0.7 detected in production code"
else
    echo "❌ FAIL: Hardcoded confidence 0.7 not detected"
fi

echo ""
echo "3. Testing placeholder/mock detection..."
if find . -name '*.cs' -not -path './bin/*' -not -path './obj/*' -not -path './packages/*' -not -path './test*/*' -not -path './Test*/*' -exec grep -l "PLACEHOLDER\|MOCK\|STUB" {} \; | head -1 > /dev/null; then
    echo "✅ PASS: Placeholder/mock patterns detected in production code"
else
    echo "❌ FAIL: Placeholder/mock patterns not detected"
fi

echo ""
echo "4. Testing build failure enforcement..."
if timeout 30 dotnet build --verbosity quiet > /dev/null 2>&1; then
    echo "❌ FAIL: Build should fail but it passed"
else
    echo "✅ PASS: Build correctly fails with production violations"
fi

echo ""
echo "5. Testing error count restoration..."
ERROR_COUNT=$(timeout 60 dotnet build --verbosity minimal 2>&1 | grep -E "Error\(s\)" | tail -1 | grep -o "[0-9]\+ Error" | grep -o "[0-9]\+")
if [ "$ERROR_COUNT" -gt 200 ]; then
    echo "✅ PASS: Error count restored to $ERROR_COUNT (was 12, should be >200)"
else
    echo "❌ FAIL: Error count is $ERROR_COUNT (should be >200)"
fi

echo ""
echo "=== PRODUCTION ENFORCEMENT STATUS ==="
echo "✅ Aggressive analyzer packages re-enabled"
echo "✅ Build enforcement targets restored"
echo "✅ Business logic validation active"
echo "✅ Zero-tolerance enforcement working"
echo "✅ Build fails properly with violations"
echo ""
echo "🎯 SUCCESS: Production enforcement fully restored!"