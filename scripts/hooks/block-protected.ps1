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
