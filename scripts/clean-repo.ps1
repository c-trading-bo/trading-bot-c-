# Safe cleanup: remove build artifacts, test outputs, and empty directories found by scan
# This script only targets items not required for bot logic or login.
# - Removes all **/bin and **/obj directories
# - Removes empty directories (after artifact removal)
# - Keeps src/, scripts/, appsettings.json, .env*, state/, journal/, tests/ source, docs/examples
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\clean-repo.ps1

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

# 1) Remove all bin/ and obj/ directories (build artifacts)
$artifactDirs = Get-ChildItem -Path . -Recurse -Force -Directory |
    Where-Object { $_.FullName -match "\\(bin|obj)$" }

foreach ($d in $artifactDirs) { Remove-DirSafe $d.FullName }

# 2) Remove known empty directories from previous scans and any new empties after cleanup
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

Write-Host "[clean-repo] Cleanup complete."