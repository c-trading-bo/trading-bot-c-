# Auto-pull script for keeping your local repo in sync with GitHub main branch
# Usage: powershell -ExecutionPolicy Bypass -File .\auto-pull.ps1

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$intervalSeconds = 60  # Check every 60 seconds

Write-Host "[auto-pull] Monitoring for new commits on 'main'... Press Ctrl+C to stop."

while ($true) {
    Set-Location $repoPath
    git fetch origin main
    $local = git rev-parse main
    $remote = git rev-parse origin/main
    if ($local -ne $remote) {
        Write-Host "[auto-pull] New commits detected. Pulling..."
        git pull origin main
        Write-Host "[auto-pull] Repo updated!"
    }
    else {
        Write-Host "[auto-pull] No new commits."
    }
    Start-Sleep -Seconds $intervalSeconds
}
