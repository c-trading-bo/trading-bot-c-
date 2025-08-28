param(
    [string]$Symbols = "ES",
    [string]$Strategies = "S2,S3,S6,S11",
    [int]$Days = 5,
    [switch]$SummaryOnly
)

# Ensure we execute from repo root
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

# Environment for tuner
$env:RUN_TUNING = '1'
$env:TUNE_SUMMARY_ONLY = if ($SummaryOnly) { '1' } else { '0' }
$env:TUNE_LOOKBACK_DAYS = "$Days"
$env:TOPSTEPX_SYMBOLS = $Symbols
$env:STRATEGIES = $Strategies
$env:TUNE_SYMBOLS = $Symbols
$env:TUNE_STRATEGIES = $Strategies
$env:TUNE_LOOKBACK_DAYS = "$Days"
$env:LOOKBACK_DAYS = "$Days"
$env:S3_DEBUG_REASONS = '1'
$env:SKIP_TIME_WINDOWS = '1'
$env:ALL_HOURS_QUALITY = '1'
$env:SKIP_ATTEMPT_CAPS = '1'
$env:QTH_NIGHT = '0.25'
$env:QTH_OPEN = '0.25'
$env:QTH_RTH = '0.25'
$env:TUNE_RELAX = '1'

# Try to apply tuned S3 config if present
$tunedPath = Join-Path (Get-Location) 'state\tuning\S3-StrategyConfig.tuned.json'
if (Test-Path $tunedPath) { $env:S3_CONFIG_PATH = $tunedPath }
$env:SKIP_MODE_PROMPT = '1'
# Allow login if credentials are available
if (-not [string]::IsNullOrWhiteSpace($env:TOPSTEPX_USERNAME) -and -not [string]::IsNullOrWhiteSpace($env:TOPSTEPX_API_KEY)) {
    $env:AUTH_ALLOW = '1'
}

Write-Host "[Run-Tuner] Symbols=$Symbols Strategies=$Strategies Days=$Days SummaryOnly=$($SummaryOnly.IsPresent)"

# Build and run orchestrator tuner (no stale binaries)
& dotnet build .\src\OrchestratorAgent\OrchestratorAgent.csproj -c Release --nologo
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
& dotnet run -c Release --project .\src\OrchestratorAgent\OrchestratorAgent.csproj -- --days $Days

Pop-Location | Out-Null
