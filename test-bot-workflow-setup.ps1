#!/usr/bin/env pwsh
# test-bot-setup.ps1 - Verify bot.py and workflow prerequisites

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Topstep Bot Setup Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Test 1: Check Python 3.11
Write-Host "1. Checking Python 3.11..." -ForegroundColor Yellow
try {
    $pythonVersion = python3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Python 3.11 not found" -ForegroundColor Red
        Write-Host "   Install: choco install python311 -y" -ForegroundColor Yellow
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ Python 3.11 not found: $_" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 2: Check .NET SDK
Write-Host "2. Checking .NET SDK..." -ForegroundColor Yellow
try {
    $dotnetVersion = dotnet --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ .NET SDK v$dotnetVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ .NET SDK not found" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ .NET SDK not found: $_" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 3: Check pip
Write-Host "3. Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = python3.11 -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ pip not found" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ pip not found: $_" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 4: Check project-x-py SDK
Write-Host "4. Checking project-x-py SDK..." -ForegroundColor Yellow
try {
    $sdkCheck = python3.11 -c "import project_x_py; print(f'v{project_x_py.__version__}')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ project-x-py $sdkCheck" -ForegroundColor Green
    } else {
        Write-Host "   ❌ project-x-py not installed" -ForegroundColor Red
        Write-Host "   Install: python3.11 -m pip install 'project-x-py[all]>=3.5.0'" -ForegroundColor Yellow
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ project-x-py not installed: $_" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 5: Check bot.py exists
Write-Host "5. Checking bot.py..." -ForegroundColor Yellow
if (Test-Path "bot.py") {
    Write-Host "   ✅ bot.py exists" -ForegroundColor Green
    
    # Test help
    try {
        $helpOutput = python3.11 bot.py --help 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ bot.py --help works" -ForegroundColor Green
        } else {
            Write-Host "   ❌ bot.py --help failed" -ForegroundColor Red
            $allGood = $false
        }
    } catch {
        Write-Host "   ❌ bot.py execution error: $_" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host "   ❌ bot.py not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 6: Check environment variables
Write-Host "6. Checking environment variables..." -ForegroundColor Yellow
$requiredVars = @("TOPSTEPX_API_KEY", "TOPSTEPX_USERNAME")
$optionalVars = @("TOPSTEPX_ACCOUNT_ID")

foreach ($var in $requiredVars) {
    if ($env:$var) {
        Write-Host "   ✅ $var is set" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $var is not set (REQUIRED)" -ForegroundColor Red
        $allGood = $false
    }
}

foreach ($var in $optionalVars) {
    if ($env:$var) {
        Write-Host "   ✅ $var is set" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️  $var is not set (optional)" -ForegroundColor Yellow
    }
}
Write-Host ""

# Test 7: Check UnifiedOrchestrator
Write-Host "7. Checking UnifiedOrchestrator..." -ForegroundColor Yellow
if (Test-Path "src/UnifiedOrchestrator/UnifiedOrchestrator.csproj") {
    Write-Host "   ✅ UnifiedOrchestrator project found" -ForegroundColor Green
} else {
    Write-Host "   ❌ UnifiedOrchestrator project not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 8: Check workflow file
Write-Host "8. Checking workflow file..." -ForegroundColor Yellow
if (Test-Path ".github/workflows/topstep-bot.yml") {
    Write-Host "   ✅ topstep-bot.yml exists" -ForegroundColor Green
    
    # Validate YAML syntax (requires PyYAML)
    try {
        $yamlCheck = python3.11 -c "import yaml; yaml.safe_load(open('.github/workflows/topstep-bot.yml'))" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ YAML syntax is valid" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Could not validate YAML (PyYAML not installed)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠️  Could not validate YAML: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ❌ topstep-bot.yml not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "  ✅ All prerequisites met!" -ForegroundColor Green
    Write-Host "" 
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Add GitHub Secrets (if not already done):" -ForegroundColor White
    Write-Host "     - TOPSTEPX_API_KEY" -ForegroundColor Yellow
    Write-Host "     - TOPSTEPX_USERNAME" -ForegroundColor Yellow
    Write-Host "  2. Configure self-hosted runner" -ForegroundColor White
    Write-Host "  3. Test locally:" -ForegroundColor White
    Write-Host "     python3.11 bot.py --dry-run" -ForegroundColor Yellow
    Write-Host "  4. Run workflow via GitHub Actions UI" -ForegroundColor White
} else {
    Write-Host "  ❌ Some prerequisites are missing" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please fix the issues above before proceeding." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

exit $(if ($allGood) { 0 } else { 1 })
