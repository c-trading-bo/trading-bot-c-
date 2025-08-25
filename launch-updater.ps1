# Load .env.local variables and launch the Updater sidecar
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

Write-Host "Launching UpdaterAgent..."
dotnet run --project src\UpdaterAgent\UpdaterAgent.csproj -c Release
