# Ensure we are at repository root (script's directory)
Set-Location -Path $PSScriptRoot

# Load .env.local variables and launch the bot
$envFile = ".env.local"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([A-Za-z0-9_]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2]
            # Respect variables already set in the current shell; never override BOT_MODE or SKIP_MODE_PROMPT
            $existing = [Environment]::GetEnvironmentVariable($name)
            $isProtected = $name -in @('BOT_MODE','SKIP_MODE_PROMPT')
            # Always force BOT_QUICK_EXIT from .env.local to prevent exit issues
            $isForced = $name -eq 'BOT_QUICK_EXIT'
            if ($isForced -or (-not $isProtected -and [string]::IsNullOrWhiteSpace($existing))) {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
    Write-Host "Loaded environment variables from $envFile."
}
else {
    Write-Host "$envFile not found. Skipping env load."
}

# Defaults (safe): prefer persistent run and a single HTTP port
# FORCE BOT_QUICK_EXIT to 0 to prevent exit issues (override any system setting)
$env:BOT_QUICK_EXIT = "0"
if (-not $env:ASPNETCORE_URLS) { $env:ASPNETCORE_URLS = "http://localhost:5050" }
if (-not $env:APP_CONCISE_CONSOLE) { $env:APP_CONCISE_CONSOLE = "true" }
if (-not $env:RUN_LEARNING) { $env:RUN_LEARNING = "1" }           # Always run the learner loop
if (-not $env:INSTANT_APPLY) { $env:INSTANT_APPLY = "1" }         # Auto-apply ParamStore overrides
if (-not $env:INSTANT_ALLOW_LIVE) { $env:INSTANT_ALLOW_LIVE = "0" } # Do NOT auto-apply while LIVE by default
if (-not $env:SKIP_MODE_PROMPT) { $env:SKIP_MODE_PROMPT = "1" }    # Default to non-interactive launch
if (-not $env:BOT_MODE) { $env:BOT_MODE = "paper" }               # Default to paper mode for safety
if (-not $env:AUTH_ALLOW) { $env:AUTH_ALLOW = "1" }               # Allow login if username/key provided
if (-not $env:TOPSTEPX_API_BASE)   { $env:TOPSTEPX_API_BASE   = "https://api.topstepx.com" }
if (-not $env:TOPSTEPX_RTC_BASE)   { $env:TOPSTEPX_RTC_BASE   = "https://rtc.topstepx.com" }
if (-not $env:RTC_USER_HUB)        { $env:RTC_USER_HUB        = "https://rtc.topstepx.com/hubs/user" }
if (-not $env:RTC_MARKET_HUB)      { $env:RTC_MARKET_HUB      = "https://rtc.topstepx.com/hubs/market" }
if (-not $env:AUTO_GO_LIVE)        { $env:AUTO_GO_LIVE        = "false" }
if (-not $env:AUTO_STICKY_LIVE)    { $env:AUTO_STICKY_LIVE    = "false" }
if (-not $env:AUTO_MIN_HEALTHY_PASSES) { $env:AUTO_MIN_HEALTHY_PASSES = "3" }

# Optional interactive mode selection (Live/Paper/Shadow)
$skipPromptRaw = $env:SKIP_MODE_PROMPT
$skipPrompt = $false
if ($skipPromptRaw) { $skipPrompt = $skipPromptRaw.Trim().ToLowerInvariant() -in @("1", "true", "yes") }
if (-not $skipPrompt) {
    Write-Host ""
    Write-Host "Select mode:"
    Write-Host "  [L]ive   (places real orders)"
    Write-Host "  [P]aper  (simulates orders)"
    Write-Host "  [S]hadow (no orders)"
    Write-Host "  [P]aper  (simulates orders) [default]"
    $choice = Read-Host -Prompt "Mode"
    if ([string]::IsNullOrWhiteSpace($choice)) { $lower = "p" } else { $lower = $choice.Trim().ToLowerInvariant() }
    switch ($lower) {
        "l" { $env:BOT_MODE = "live" }
        "live" { $env:BOT_MODE = "live" }
        "p" { $env:BOT_MODE = "paper" }
        "paper" { $env:BOT_MODE = "paper" }
        default { $env:BOT_MODE = "shadow" }
    }
    # Prevent a second prompt inside the app
    $env:SKIP_MODE_PROMPT = "1"
    Write-Host "Mode selected: $env:BOT_MODE"
}

if ($env:BOT_QUICK_EXIT -eq "1") {
    Write-Host "Quick-exit mode enabled (BOT_QUICK_EXIT=1). Orchestrator will start and exit after a short delay."
}

# Optional: skip probe (accept 1|true|yes case-insensitive)
$skipProbeRaw = $env:SKIP_CONNECTIVITY_PROBE
$skipProbe = $false
if ($skipProbeRaw) {
    $skipProbe = $skipProbeRaw.Trim().ToLowerInvariant() -in @("1", "true", "yes")
}
if ($skipProbe) {
    Write-Host "Skipping connectivity probe (SKIP_CONNECTIVITY_PROBE=$skipProbeRaw)."
}
else {
    Write-Host "Running connectivity probe..."
    $probePath = Join-Path $PSScriptRoot 'ConnectivityProbe'
    if (Test-Path $probePath) {
        dotnet run --project ConnectivityProbe
        if ($LASTEXITCODE -eq 1) {
            Write-Host "Connectivity probe failed (transport/network error). Continuing (non-fatal)."
        }
        elseif ($LASTEXITCODE -eq 2) {
            Write-Host "Connectivity probe: missing JWT/login credentials. Continuing to launch."
        }
        else {
            Write-Host "Connectivity probe passed."
        }
    }
    else {
        Write-Host "Connectivity probe project not found. Skipping."
    }
}

# Single-port guard: avoid double launches on the same HTTP port (LISTEN only)
try {
    try {
        $uri = [Uri]$env:ASPNETCORE_URLS
        $port = $uri.Port
    }
    catch { $port = 5050 }

    $listener = $null
    try {
        $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop
    }
    catch { $listener = $null }

    if (-not $listener) {
        # Fallback: netstat LISTENING line for :$port
        $ns = netstat -ano | Select-String -Pattern ("LISTENING.*:$port\b") | Select-Object -First 1
        if ($ns) { $listener = $true }
    }

    if ($listener) {
        Write-Host "Port $port is already listening. If another bot is running, close it or change ASPNETCORE_URLS."
        exit 1
    }
}
catch { }

# Launch the bot
Write-Host "Launching bot on $env:ASPNETCORE_URLS ..."
dotnet run --project src\OrchestratorAgent\OrchestratorAgent.csproj
Write-Host "Bot process exited with code $LASTEXITCODE."
