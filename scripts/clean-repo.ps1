# Safe cleanup: remove build artifacts, reports, and empty directories
# This script only targets items not required for bot logic or runtime.
# - Removes all **/bin and **/obj directories
# - Removes SCAN_REPORT.txt and SCAN_STUBS.txt (generated reports)
# - Optionally removes examples/, ConnectivityProbe, TopstepAI.system.md, src/SimulationAgent (use -Aggressive)
# - Removes empty directories (after artifact removal)
# - Keeps src/, scripts/, appsettings.json, .env*, state/, journal/, tests/ source, docs by default
# Usage:
#   - Safe default:   powershell -ExecutionPolicy Bypass -File .\scripts\clean-repo.ps1
#   - Aggressive:     powershell -ExecutionPolicy Bypass -File .\scripts\clean-repo.ps1 -Aggressive

param(
    [switch]$Aggressive
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) | Out-Null
Set-Location -Path (Resolve-Path '..')

function Remove-DirSafe($dir) {
    try {
        if (Test-Path -LiteralPath $dir) {
            Write-Host "[clean-repo] Removing directory: $dir"
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop
        }
    } catch {
        Write-Warning "[clean-repo] Failed to remove '$dir': $($_.Exception.Message)"
    }
}
function Remove-FileSafe($file) {
    try {
        if (Test-Path -LiteralPath $file) {
            Write-Host "[clean-repo] Removing file: $file"
            Remove-Item -LiteralPath $file -Force -ErrorAction Stop
        }
    } catch {
        Write-Warning "[clean-repo] Failed to remove '$file': $($_.Exception.Message)"
    }
}

# 1) Remove all bin/ and obj/ directories (build artifacts)
$artifactDirs = Get-ChildItem -Path . -Recurse -Force -Directory |
    Where-Object { $_.FullName -match "\\(bin|obj)$" }
foreach ($d in $artifactDirs) { Remove-DirSafe $d.FullName }

# 2) Remove generated scan reports (safe)
Remove-FileSafe "SCAN_REPORT.txt"
Remove-FileSafe "SCAN_STUBS.txt"

# 3) Optional aggressive removals: developer-only and scaffolding not needed to run the bot
if ($Aggressive) {
    $optDirs = @(
        "examples",
        "ConnectivityProbe",
        "src\SimulationAgent"
    )
    foreach ($opt in $optDirs) { if (Test-Path -LiteralPath $opt) { Remove-DirSafe $opt } }

    $optFiles = @(
        "TopstepAI.system.md"
    )
    foreach ($f in $optFiles) { Remove-FileSafe $f }
}

# 4) Remove empty directories after cleanup
function Get-EmptyDirs {
    $dirs = Get-ChildItem -Path . -Recurse -Force -Directory | Where-Object { $_.FullName -notmatch "\\.git\\|\\.vs\\" }
    $empties = @()
    foreach ($dir in $dirs) {
        try {
            $hasFile = (Get-ChildItem -Path $dir.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Select-Object -First 1)
            if (-not $hasFile) { $empties += $dir.FullName }
        } catch { }
    }
    return $empties
}

$empties = Get-EmptyDirs
foreach ($e in $empties) { Remove-DirSafe $e }

Write-Host "[clean-repo] Cleanup complete. Aggressive=$($Aggressive.IsPresent)"