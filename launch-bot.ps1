# Load .env.local variables and launch the bot
$envFile = ".env.local"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([A-Za-z0-9_]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2]
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    Write-Host "Loaded environment variables from $envFile."
} else {
    Write-Host "$envFile not found. Skipping env load."
}

# Optional: skip probe
$skipProbe = $env:SKIP_CONNECTIVITY_PROBE
if (-not $skipProbe) {
    Write-Host "Running connectivity probe..."
    dotnet run --project ConnectivityProbe
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Connectivity probe failed (exit code $LASTEXITCODE). Set TOPSTEPX_JWT or TOPSTEPX_USERNAME/API_KEY and try again."
        exit $LASTEXITCODE
    }
}

# Launch the bot
Write-Host "Launching bot..."
dotnet run --project src\OrchestratorAgent\OrchestratorAgent.csproj
