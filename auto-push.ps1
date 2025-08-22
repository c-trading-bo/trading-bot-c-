# Auto-push script for automatically committing and pushing all changes
# Usage: powershell -ExecutionPolicy Bypass -File .\auto-push.ps1

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$intervalSeconds = 60  # Check every 60 seconds

Write-Host "[auto-push] Monitoring for file changes... Press Ctrl+C to stop."

while ($true) {
    Set-Location $repoPath
    $status = git status --porcelain
    if ($status) {
        git add -A
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit -m "Auto-push: $timestamp"
        git push origin main
        Write-Host "[auto-push] Changes pushed at $timestamp."
    } else {
        Write-Host "[auto-push] No changes detected."
    }
    Start-Sleep -Seconds $intervalSeconds
}
