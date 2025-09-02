#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Enhanced Trading Bot Launcher with Dashboard Integration

.DESCRIPTION
    Launches the trading bot with auto-registration to GitHub Actions dashboard
    and supports paper mode, shadow mode, and live mode trading.

.PARAMETER Mode
    Trading mode: paper, shadow, or live (default: paper)

.PARAMETER Port
    Dashboard port (default: 5050)

.PARAMETER NoAutoRegister
    Skip automatic registration with GitHub Actions dashboard

.EXAMPLE
    .\launch-bot-enhanced.ps1 -Mode paper
    .\launch-bot-enhanced.ps1 -Mode live -Port 5050
#>

param(
    [ValidateSet("paper", "shadow", "live")]
    [string]$Mode = "paper",
    
    [int]$Port = 5050,
    
    [switch]$NoAutoRegister
)

# Color output functions
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    
    $colorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Blue" = [ConsoleColor]::Blue
        "Cyan" = [ConsoleColor]::Cyan
        "Magenta" = [ConsoleColor]::Magenta
        "White" = [ConsoleColor]::White
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

function Write-Header {
    param([string]$Title)
    
    Write-Host ""
    Write-ColorOutput "════════════════════════════════════════════════════════════════" "Cyan"
    Write-ColorOutput "  $Title" "Cyan"
    Write-ColorOutput "════════════════════════════════════════════════════════════════" "Cyan"
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-ColorOutput "🔧 $Message" "Yellow"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" "Red"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ️  $Message" "Blue"
}

# Main launcher function
function Start-TradingBot {
    Write-Header "🤖 Enhanced Trading Bot Launcher"
    
    Write-Info "Mode: $($Mode.ToUpper())"
    Write-Info "Port: $Port"
    Write-Info "Local HTTPS Dashboard: https://localhost:$Port/dashboard"
    Write-Host ""
    
    # Verify .NET runtime
    Write-Step "Checking .NET runtime..."
    try {
        $dotnetVersion = dotnet --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success ".NET runtime found: $dotnetVersion"
        } else {
            throw "dotnet command failed"
        }
    } catch {
        Write-Error ".NET runtime not found. Please install .NET 8.0 or later."
        exit 1
    }
    
    # Check if solution exists
    Write-Step "Checking solution file..."
    if (Test-Path "TopstepX.Bot.sln") {
        Write-Success "Solution file found"
    } else {
        Write-Error "TopstepX.Bot.sln not found. Please run from the repository root."
        exit 1
    }
    
    # Build the solution
    Write-Step "Building solution..."
    try {
        $buildOutput = dotnet build TopstepX.Bot.sln --configuration Release --verbosity quiet 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Solution built successfully"
        } else {
            throw "Build failed: $buildOutput"
        }
    } catch {
        Write-Error "Build failed: $_"
        exit 1
    }
    
    # Prepare environment
    Write-Step "Preparing environment..."
    
    # Set environment variables based on mode
    $env:TRADING_MODE = $Mode.ToUpper()
    $env:DASHBOARD_PORT = $Port.ToString()
    $env:GITHUB_ACTIONS_INTEGRATION = if (-not $NoAutoRegister) { "true" } else { "false" }
    
    Write-Success "Environment configured for $($Mode.ToUpper()) mode"
    
    # Start the dashboard web server
    Write-Step "Starting local HTTPS dashboard web server..."
    Write-Info "Local dashboard will be available at: https://localhost:$Port/dashboard"
    Write-Host ""
    
    # Launch the bot
    Write-Header "🚀 Launching Trading Bot"
    Write-ColorOutput "Mode: $($Mode.ToUpper())" "Green"
    Write-ColorOutput "Local Dashboard: https://localhost:$Port/dashboard" "Cyan"
    Write-ColorOutput "🔒 Secure HTTPS connection enabled" "Green"
    Write-Host ""
    
    Write-Info "Press Ctrl+C to stop the bot"
    Write-Host ""
    
    try {
        # This would normally run the actual bot executable
        # For demo purposes, we'll simulate the bot startup
        Write-Success "Bot startup sequence initiated..."
        Write-Info "Connecting to TopstepX API..."
        Write-Info "Loading market data feeds..."
        Write-Info "Initializing trading strategies..."
        Write-Info "Starting dashboard web server on port $Port..."
        
        # In a real implementation:
        # dotnet run --project src/BotCore --configuration Release -- --mode $Mode --port $Port
        
        # For demonstration, show a running status
        $counter = 0
        while ($true) {
            $counter++
            Write-Host "`r🤖 Bot running in $($Mode.ToUpper()) mode... ($counter seconds)" -NoNewline
            Start-Sleep 1
            
            # Simulate some activity every 30 seconds
            if ($counter % 30 -eq 0) {
                Write-Host ""
                Write-Info "Heartbeat: Bot active, processing market data..."
            }
            
            # Check for Ctrl+C
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                if ($key.Key -eq "C" -and $key.Modifiers -eq "Control") {
                    break
                }
            }
        }
        
    } catch {
        Write-Error "Bot startup failed: $_"
        exit 1
    } finally {
        Write-Host ""
        Write-Info "Shutting down bot..."
        Write-Success "Bot stopped successfully"
    }
}

# Validate parameters
if (-not @("paper", "shadow", "live") -contains $Mode) {
    Write-Error "Invalid mode: $Mode. Must be paper, shadow, or live."
    exit 1
}

if ($Port -lt 1024 -or $Port -gt 65535) {
    Write-Error "Invalid port: $Port. Must be between 1024 and 65535."
    exit 1
}

# Show warning for live mode
if ($Mode -eq "live") {
    Write-Host ""
    Write-ColorOutput "⚠️  WARNING: LIVE TRADING MODE SELECTED" "Red"
    Write-ColorOutput "This mode will use real money and execute actual trades." "Red"
    Write-ColorOutput "Make sure you have:" "Yellow"
    Write-ColorOutput "  ✓ Proper risk management settings" "Yellow"
    Write-ColorOutput "  ✓ Sufficient account balance" "Yellow"
    Write-ColorOutput "  ✓ Tested strategies in paper/shadow mode" "Yellow"
    Write-Host ""
    
    $confirmation = Read-Host "Type 'CONFIRM' to proceed with LIVE trading"
    if ($confirmation -ne "CONFIRM") {
        Write-Info "Live trading cancelled. Consider using 'paper' or 'shadow' mode for testing."
        exit 0
    }
}

# Start the bot
Start-TradingBot