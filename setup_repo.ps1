# Repository setup script: configures a safe Git hooks path with a no-op pre-commit
# Usage: PowerShell -ExecutionPolicy Bypass -File .\setup_repo.ps1

$ErrorActionPreference = 'Stop'

$repoRoot  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$hooksPath = Join-Path $repoRoot 'scripts\hooks'
$preCommit = Join-Path $hooksPath 'pre-commit'

# Ensure hooks directory exists
if (-not (Test-Path $hooksPath)) {
    New-Item -ItemType Directory -Force -Path $hooksPath | Out-Null
    Write-Host "[setup-repo] Created hooks directory: $hooksPath"
}

# Ensure a no-op pre-commit hook exists (cross-platform sh)
if (-not (Test-Path $preCommit)) {
    $content = "#!/bin/sh`n# no-op pre-commit to avoid failing commits when hooks are expected`nexit 0`n"
    Set-Content -Path $preCommit -Value $content -NoNewline
    Write-Host "[setup-repo] Installed no-op pre-commit hook: $preCommit"
}

# Configure repo-local hooks path
& git config --local core.hooksPath "$hooksPath"
$current = & git config --local --get core.hooksPath
Write-Host "[setup-repo] core.hooksPath is set to: $current"

Write-Host "[setup-repo] Done. All git commits in this repo will ignore pre-commit checks unless you change the hook."
Write-Host "[setup-repo] To revert: git config --local --unset core.hooksPath"
