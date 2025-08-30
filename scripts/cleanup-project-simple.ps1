#!/usr/bin/env pwsh
# Project Cleanup Script - Remove dead code and unused projects

Write-Host "Starting comprehensive project cleanup..." -ForegroundColor Cyan

$RootPath = (Get-Location).Path
$Removed = @()
$Errors = @()

function Remove-SafelyWithLog {
    param([string]$Path, [string]$Description)
    
    if (Test-Path $Path) {
        try {
            Remove-Item $Path -Recurse -Force -ErrorAction Stop
            $Removed += "$Description`: $Path"
            Write-Host "Removed: $Description" -ForegroundColor Green
        }
        catch {
            $Errors += "$Description`: $($_.Exception.Message)"
            Write-Host "Failed to remove $Description`: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "Not found: $Description at $Path" -ForegroundColor Yellow
    }
}

# 1. Remove duplicate project directories
Write-Host "Removing duplicate project directories..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\trading-bot-c-" "Duplicate project directory (trading-bot-c-)"
Remove-SafelyWithLog "$RootPath\trading-bot-c--1" "Duplicate project directory (trading-bot-c--1)"

# 2. Remove unused standalone agent projects
Write-Host "Removing unused agent projects..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\src\LogAgent" "Unused LogAgent project"
Remove-SafelyWithLog "$RootPath\src\OrderRouterAgent" "Unused OrderRouterAgent project"
Remove-SafelyWithLog "$RootPath\src\PositionAgent" "Unused PositionAgent project"
Remove-SafelyWithLog "$RootPath\src\RiskAgent" "Unused RiskAgent project"
Remove-SafelyWithLog "$RootPath\src\UserHubAgent" "Unused UserHubAgent project"

# 3. Clean up excessive setup history files
Write-Host "Cleaning setup history files..." -ForegroundColor Yellow
$SetupFiles = Get-ChildItem "$RootPath\state\setup\setup-*.json" | Sort-Object Name -Descending
if ($SetupFiles.Count -gt 5) {
    $FilesToRemove = $SetupFiles | Select-Object -Skip 5
    foreach ($file in $FilesToRemove) {
        Remove-SafelyWithLog $file.FullName "Old setup file ($($file.Name))"
    }
}

# 4. Remove old backup contents
Write-Host "Cleaning backup artifacts..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\backups\backup-20250828-203100" "Extracted backup folder"

# 5. Remove obsolete launch scripts
Write-Host "Removing obsolete scripts..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\start-clean.sh" "Obsolete bash script"
Remove-SafelyWithLog "$RootPath\start-clean.cmd" "Obsolete batch script"

# 6. Clean up env backup files
Write-Host "Cleaning environment file backups..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\.env.local.bak" "Environment backup file"

# 7. Remove obsolete documentation files
Write-Host "Removing obsolete docs..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\S2-Strategy-TODO.md" "Obsolete strategy todo (S2)"
Remove-SafelyWithLog "$RootPath\S3-Strategy-TODO.md" "Obsolete strategy todo (S3)"

# 8. Clean up IDE-specific folders
Write-Host "Cleaning IDE artifacts..." -ForegroundColor Yellow
Remove-SafelyWithLog "$RootPath\.idea" "JetBrains IDE folder"

# Summary Report
Write-Host "Cleanup Summary:" -ForegroundColor Cyan
Write-Host "Successfully removed $($Removed.Count) items" -ForegroundColor Green
Write-Host "Failed to remove $($Errors.Count) items" -ForegroundColor Red

if ($Removed.Count -gt 0) {
    Write-Host "Removed items:" -ForegroundColor Green
    foreach ($item in $Removed) {
        Write-Host "  * $item"
    }
}

if ($Errors.Count -gt 0) {
    Write-Host "Errors encountered:" -ForegroundColor Red
    foreach ($error in $Errors) {
        Write-Host "  * $error"
    }
}

Write-Host "Project cleanup completed!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  * Run 'dotnet build' to verify all references are intact"
Write-Host "  * Consider running 'dotnet format' to standardize code formatting"
Write-Host "  * Update .gitignore if needed to prevent future clutter"
