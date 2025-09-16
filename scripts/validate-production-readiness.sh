#!/bin/bash

# ===================================
# PRODUCTION READINESS VALIDATOR
# ===================================
# Run this script before deploying to production

echo "🔍 PRODUCTION READINESS VALIDATION"
echo "=================================="

# Check for hardcoded credentials
echo "🔐 Checking for hardcoded credentials..."
if grep -r "TOPSTEPX_API_KEY.*=" .env 2>/dev/null | grep -v "YOUR_.*_HERE" | grep -v "template" | grep -v "example"; then
    echo "❌ SECURITY RISK: Hardcoded credentials found in .env files"
    exit 1
fi

if grep -r "ghp_\|github_pat_" src/ 2>/dev/null; then
    echo "❌ SECURITY RISK: GitHub tokens found in source code"
    exit 1
fi

# Check configuration
echo "📋 Checking production configuration..."
if grep -q '"ClientType": "Mock"' src/UnifiedOrchestrator/appsettings.json; then
    echo "❌ CONFIGURATION ERROR: Mock client still enabled"
    exit 1
fi

if grep -q '"AllowMockData": true' src/UnifiedOrchestrator/appsettings.json; then
    echo "❌ CONFIGURATION ERROR: Mock data still allowed"
    exit 1
fi

# Check for empty catch blocks
echo "🛡️ Checking error handling..."
if find src/ -name "*.cs" -exec grep -l "catch.*{.*}" {} \; | head -1 >/dev/null; then
    echo "⚠️  WARNING: Empty catch blocks found - review error handling"
fi

# Check for blocking async calls
echo "⚡ Checking async patterns..."
if find src/ -name "*.cs" -exec grep -l "\.Result\|\.Wait()" {} \; | head -1 >/dev/null; then
    echo "⚠️  WARNING: Blocking async calls found - may cause deadlocks"
fi

# Check SSL configuration
echo "🔒 Checking SSL configuration..."
if ! grep -q "ENABLE_SSL_VALIDATION=true" .env.production-secure; then
    echo "⚠️  WARNING: SSL validation should be enabled for production"
fi

# Build verification
echo "🏗️ Verifying build..."
if ! dotnet build --configuration Release --verbosity quiet; then
    echo "❌ BUILD FAILED: Fix compilation errors before production deployment"
    exit 1
fi

echo "✅ PRODUCTION READINESS VALIDATION COMPLETE"
echo ""
echo "🚀 Ready for production deployment with:"
echo "   • Real TopstepX API integration"
echo "   • Secure credential management"
echo "   • Production error handling"
echo "   • Clean build verification"
echo ""
echo "📋 Next steps:"
echo "   1. Set environment variables: TOPSTEPX_API_KEY, TOPSTEPX_USERNAME, TOPSTEPX_ACCOUNT_ID"
echo "   2. Configure SSL certificates"
echo "   3. Set up monitoring and logging"
echo "   4. Deploy to production environment"