# PowerShell setup script for Cline TopstepX bot
# Installs Cline, writes .env.local, and opens VS Code

# Install Cline globally (if not already installed)
if (-not (Get-Command cline -ErrorAction SilentlyContinue)) {
    npm install -g cline
}

# Create .env.local with placeholder credentials
$envPath = Join-Path $PSScriptRoot ".env.local"
if (-not (Test-Path $envPath)) {
    @"
TSX_USERNAME=you@example.com
TSX_API_KEY=your_projectx_api_key
"@ | Set-Content $envPath
}

# Open VS Code in this folder
code $PSScriptRoot

Write-Host "Cline setup complete. Edit .env.local with your credentials."
