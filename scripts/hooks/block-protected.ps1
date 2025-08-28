Param()

# Block commits that modify protected paths. Override with ALLOW_PROTECTED_EDITS=1
$ErrorActionPreference = 'Stop'

function Get-StagedFiles {
    $out = git diff --cached --name-only 2>$null
    if (-not $out) { return @() }
    return $out -split "`n" | Where-Object { $_ -ne '' }
}

$files = Get-StagedFiles
if (-not $files -or $files.Count -eq 0) {
    exit 0
}

# Case-insensitive regexes for protected areas
$protected = @(
    # Secrets / env / keys
    '^(.*/)?\.env(\..*)?$',
    '^(secrets|keys)(/|\\)',
    # CI workflows
    '^\.github(/|\\)workflows(/|\\).+\.yml$',
    # Auth/login code
    '^(src(/|\\))?TopstepAuthAgent(/|\\)',
    '^(src(/|\\))?BotCore(/|\\)Auth(/|\\)',
    '^(src(/|\\))?BotCore(/|\\)JwtCache\.cs$',
    # Risk code
    '^(src(/|\\))?RiskAgent(/|\\)',
    '^(src(/|\\))?BotCore(/|\\)Risk(/|\\)'
)

$protectedHits = @()
foreach ($f in $files) {
    foreach ($rx in $protected) {
        if ($f -imatch $rx) { $protectedHits += $f; break }
    }
}

if ($protectedHits.Count -gt 0 -and $env:ALLOW_PROTECTED_EDITS -ne '1') {
    Write-Host "\n[BLOCK] Protected files detected in commit:" -ForegroundColor Yellow
    $protectedHits | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
    Write-Host "\nThis repository blocks commits to auth/login, risk, env/keys, and CI workflows." -ForegroundColor Yellow
    Write-Host "To bypass intentionally, set ALLOW_PROTECTED_EDITS=1 for this commit and try again." -ForegroundColor Yellow
    exit 1
}

if ($protectedHits.Count -gt 0) {
    Write-Host "[WARN] Bypassing protected guard for:" -ForegroundColor Yellow
    $protectedHits | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
}

exit 0
param([string[]]$Files)

# Guardrails: always-blocked vs. launch-critical (double-approval bypass for launch-only)

# Always blocked (never bypassed): env/secrets/keys/CI/versioning/risk/auth
$protectedAlways = @(
  '^\.env(\.local)?$',
  '^keys/','^secrets/',
  '^\.github/workflows/','^azure-pipelines\.yml$','^\.gitlab-ci\.yml$',
  '^Directory\.Build\.props$','^\.version$','^version\.txt$','^CHANGELOG\.md$',
  '^src/.*/Risk/','^src/.*Risk.*\.cs$',
  '^src/TopstepAuthAgent/.*','^src/.*/Auth/.*','^src/.*/JwtCache\.cs$'
)

# Launch-critical (bypass only with maintainer approval): orchestrator entry, health host, launch scripts, solution
$protectedLaunch = @(
  '^src/OrchestratorAgent/Program\.cs$',
  '^src/OrchestratorAgent/Health/.*',
  '^launch-bot\.(ps1|cmd|sh)$','^launch-updater\.ps1$',
  '^start-clean\.(cmd|sh)$',
  '^TopstepX\.Bot\.sln$'
)

# Compute matches
$blockedAlways = @()
$blockedLaunch = @()
foreach ($f in $Files) {
  $rel = ($f -replace '\\','/')
  if ($rel -match ($protectedAlways -join '|')) { $blockedAlways += $rel; continue }
  if ($rel -match ($protectedLaunch -join '|')) { $blockedLaunch += $rel; continue }
}

# Bypass for launch-only: requires ALLOW_LAUNCH_EDITS=1|true|yes AND state/ALLOW_LAUNCH_EDITS.ok to exist
$allowEnv = [Environment]::GetEnvironmentVariable('ALLOW_LAUNCH_EDITS')
$allowFlag = $false
if (-not [string]::IsNullOrWhiteSpace($allowEnv)) {
  $flagVal = $allowEnv.Trim().ToLowerInvariant()
  if ($flagVal -in @('1','true','yes')) { $allowFlag = $true }
}
$flagPath = Join-Path (Split-Path $PSScriptRoot -Parent) '..' | Join-Path -ChildPath 'state/ALLOW_LAUNCH_EDITS.ok'
$flagExists = Test-Path $flagPath

if ($blockedAlways.Count -gt 0) {
  Write-Host "Pre-commit: blocked edits to protected files (always-blocked):" -ForegroundColor Red
  $blockedAlways | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
  Write-Host "These files are never editable via commit. Use a reviewed PR with maintainer approval to change." -ForegroundColor Yellow
  exit 1
}

if ($blockedLaunch.Count -gt 0 -and -not ($allowFlag -and $flagExists)) {
  Write-Host "Pre-commit: blocked edits to launch-critical files:" -ForegroundColor Red
  $blockedLaunch | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
  Write-Host "To allow this commit, set ALLOW_LAUNCH_EDITS=1 and create 'state/ALLOW_LAUNCH_EDITS.ok' (contains your approval note)." -ForegroundColor Yellow
  exit 1
}

exit 0
