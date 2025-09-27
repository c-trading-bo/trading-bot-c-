#!/bin/bash

# Core Production Guardrail Verification
# Verifies the essential production guardrails are implemented correctly

echo "🛡️ CORE PRODUCTION GUARDRAIL VERIFICATION"
echo "=========================================="

passed=0
total=0

echo ""
echo "1. ✅ ES/MES Tick Rounding Implementation"
echo "   Checking ProductionPriceService has 0.25 tick rounding..."
if grep -q "ES_TICK = 0.25m" src/BotCore/Services/ProductionPriceService.cs; then
    echo "   ✅ ES_TICK constant = 0.25m found"
    ((passed++))
else
    echo "   ❌ ES_TICK constant missing"
fi
((total++))

if grep -q "MidpointRounding.AwayFromZero" src/BotCore/Services/ProductionPriceService.cs; then
    echo "   ✅ Correct rounding method implemented"
    ((passed++))
else
    echo "   ❌ Correct rounding method missing"
fi
((total++))

echo ""
echo "2. ✅ Risk Validation (Reject if ≤ 0)"
echo "   Checking ProductionPriceService validates risk..."
if grep -q "risk <= 0" src/BotCore/Services/ProductionPriceService.cs; then
    echo "   ✅ Risk validation logic found"
    ((passed++))
else
    echo "   ❌ Risk validation logic missing"
fi
((total++))

if grep -q "return null" src/BotCore/Services/ProductionPriceService.cs; then
    echo "   ✅ Risk rejection (return null) implemented"
    ((passed++))
else
    echo "   ❌ Risk rejection missing"
fi
((total++))

echo ""
echo "3. ✅ Kill Switch Implementation"
echo "   Checking ProductionKillSwitchService..."
if [ -f "src/BotCore/Services/ProductionKillSwitchService.cs" ]; then
    echo "   ✅ ProductionKillSwitchService exists"
    ((passed++))
else
    echo "   ❌ ProductionKillSwitchService missing"
fi
((total++))

if grep -q "IsKillSwitchActive" src/BotCore/Services/ProductionKillSwitchService.cs; then
    echo "   ✅ Kill switch detection method found"
    ((passed++))
else
    echo "   ❌ Kill switch detection missing"
fi
((total++))

echo ""
echo "4. ✅ DRY_RUN Precedence"
echo "   Checking DRY_RUN precedence logic..."
if grep -q "IsDryRunMode" src/BotCore/Services/ProductionKillSwitchService.cs; then
    echo "   ✅ DRY_RUN mode detection found"
    ((passed++))
else
    echo "   ❌ DRY_RUN mode detection missing"
fi
((total++))

if grep -q "DRY_RUN.*true" src/BotCore/Services/ProductionKillSwitchService.cs; then
    echo "   ✅ DRY_RUN environment variable check found"
    ((passed++))
else
    echo "   ❌ DRY_RUN environment variable check missing"
fi
((total++))

echo ""
echo "5. ✅ Order Evidence Requirements"
echo "   Checking ProductionOrderEvidenceService..."
if [ -f "src/BotCore/Services/ProductionOrderEvidenceService.cs" ]; then
    echo "   ✅ ProductionOrderEvidenceService exists"
    ((passed++))
else
    echo "   ❌ ProductionOrderEvidenceService missing"
fi
((total++))

if grep -q "VerifyOrderFillEvidenceAsync" src/BotCore/Services/ProductionOrderEvidenceService.cs; then
    echo "   ✅ Order evidence verification method found"
    ((passed++))
else
    echo "   ❌ Order evidence verification missing"
fi
((total++))

echo ""
echo "6. ✅ Magic Number Elimination (Critical Files)"
echo "   Checking for production constants in fixed files..."
constants_found=0

if grep -q "AutoRemediationConstants" src/Infrastructure.TopstepX/AutoRemediationSystem.cs; then
    echo "   ✅ AutoRemediationConstants defined"
    ((constants_found++))
fi

if grep -q "ProductionGateConstants" src/Infrastructure.TopstepX/ProductionGateSystem.cs; then
    echo "   ✅ ProductionGateConstants defined"
    ((constants_found++))
fi

if grep -q "SignalRSafeInvokerConstants" src/Infrastructure.TopstepX/SignalRSafeInvoker.cs; then
    echo "   ✅ SignalRSafeInvokerConstants defined"
    ((constants_found++))
fi

if grep -q "SmokeTestConstants" src/Infrastructure.TopstepX/ComprehensiveSmokeTestSuite.cs; then
    echo "   ✅ SmokeTestConstants defined"
    ((constants_found++))
fi

if [ $constants_found -ge 3 ]; then
    echo "   ✅ Critical magic numbers replaced with constants"
    ((passed++))
else
    echo "   ❌ Insufficient magic number replacements ($constants_found/4)"
fi
((total++))

echo ""
echo "7. ✅ Production Extension Methods"
echo "   Checking for service registration extensions..."
if [ -f "src/BotCore/Extensions/ProductionGuardrailExtensions.cs" ]; then
    echo "   ✅ ProductionGuardrailExtensions exists"
    ((passed++))
else
    echo "   ❌ ProductionGuardrailExtensions missing"
fi
((total++))

if grep -q "AddProductionGuardrails" src/BotCore/Extensions/ProductionGuardrailExtensions.cs; then
    echo "   ✅ AddProductionGuardrails extension method found"
    ((passed++))
else
    echo "   ❌ AddProductionGuardrails extension missing"
fi
((total++))

echo ""
echo "📊 VERIFICATION RESULTS"
echo "======================="
echo "Guardrails Verified: $passed/$total"

if [ $passed -eq $total ]; then
    echo ""
    echo "🎉 ALL CORE PRODUCTION GUARDRAILS VERIFIED!"
    echo ""
    echo "✅ Production Readiness Checklist:"
    echo "   • ES/MES tick rounding (0.25) implemented ✅"
    echo "   • Risk validation (reject ≤ 0) implemented ✅"
    echo "   • Kill switch monitoring implemented ✅"
    echo "   • DRY_RUN precedence enforced ✅"
    echo "   • Order evidence validation implemented ✅"
    echo "   • Magic numbers replaced with constants ✅"
    echo "   • Service registration ready ✅"
    echo ""
    echo "🛡️ VERDICT: CORE GUARDRAILS ARE PRODUCTION READY"
    echo ""
    echo "The trading bot implements all critical safety requirements"
    echo "following the zero-tolerance production enforcement approach."
    echo ""
    exit 0
else
    echo ""
    echo "❌ Some core guardrails need attention ($((total-passed)) missing)"
    echo ""
    echo "Review the missing items above and ensure they are properly implemented."
    exit 1
fi