# TopstepX Contract Discovery Tool
# This script pings various TopstepX endpoints to discover available contracts

param(
    [string]$JwtToken = $env:TOPSTEPX_JWT,
    [string]$BaseUrl = "https://api.topstepx.com",
    [switch]$Verbose
)

if (-not $JwtToken) {
    Write-Host "❌ JWT token not found. Set TOPSTEPX_JWT environment variable or pass -JwtToken parameter" -ForegroundColor Red
    exit 1
}

Write-Host "🔍 TopstepX Contract Discovery Tool" -ForegroundColor Cyan
Write-Host "BaseURL: $BaseUrl" -ForegroundColor Gray
Write-Host "JWT Length: $($JwtToken.Length) characters" -ForegroundColor Gray
Write-Host ""

# Setup headers
$headers = @{
    "Authorization" = "Bearer $JwtToken"
    "Content-Type" = "application/json"
    "Accept" = "application/json"
}

function Test-Endpoint {
    param(
        [string]$Method,
        [string]$Endpoint,
        [object]$Body = $null,
        [string]$Description
    )
    
    Write-Host "📡 Testing: $Description" -ForegroundColor Yellow
    Write-Host "   Method: $Method $Endpoint" -ForegroundColor Gray
    
    try {
        $uri = "$BaseUrl$Endpoint"
        
        if ($Method -eq "GET") {
            $response = Invoke-RestMethod -Uri $uri -Method GET -Headers $headers -TimeoutSec 10
        } else {
            $bodyJson = $Body | ConvertTo-Json -Depth 10
            if ($Verbose) {
                Write-Host "   Body: $bodyJson" -ForegroundColor Gray
            }
            $response = Invoke-RestMethod -Uri $uri -Method POST -Headers $headers -Body $bodyJson -TimeoutSec 10
        }
        
        Write-Host "   ✅ SUCCESS" -ForegroundColor Green
        
        # Try to extract contract information
        $contracts = @()
        
        if ($response.data -is [array]) {
            $contracts = $response.data
        } elseif ($response.data) {
            $contracts = @($response.data)
        } elseif ($response -is [array]) {
            $contracts = $response
        } else {
            $contracts = @($response)
        }
        
        foreach ($contract in $contracts) {
            if ($contract.contractId -or $contract.id -or $contract.instrumentId) {
                $contractId = if ($contract.contractId) { $contract.contractId } elseif ($contract.id) { $contract.id } else { $contract.instrumentId }
                $symbol = if ($contract.symbol) { $contract.symbol } elseif ($contract.instrument) { $contract.instrument } elseif ($contract.name) { $contract.name } else { "Unknown" }
                Write-Host "   📋 Contract: $symbol -> ID: $contractId" -ForegroundColor Cyan
            }
        }
        
        if ($Verbose -and $contracts.Count -gt 0) {
            Write-Host "   Raw Response:" -ForegroundColor Gray
            $response | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor DarkGray
        }
        
        Write-Host ""
        return $true
        
    } catch {
        $statusCode = "Unknown"
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode
        }
        Write-Host "   ❌ FAILED: $statusCode - $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        return $false
    }
}

# Test various endpoints
Write-Host "🚀 Starting endpoint discovery..." -ForegroundColor Green
Write-Host ""

# Contract endpoints
Test-Endpoint "POST" "/api/Contract/available" @{live = $false} "Contract Available (Sim)"
Test-Endpoint "POST" "/api/Contract/available" @{live = $true} "Contract Available (Live)"
Test-Endpoint "POST" "/api/Contract/search" @{symbol = "ES"} "Contract Search (ES)"
Test-Endpoint "POST" "/api/Contract/search" @{symbol = "NQ"} "Contract Search (NQ)"
Test-Endpoint "POST" "/api/Contract/search" @{search = "ES"} "Contract Search (ES - search field)"
Test-Endpoint "POST" "/api/Contract/search" @{query = "ES"} "Contract Search (ES - query field)"

# Instrument endpoints
Test-Endpoint "POST" "/api/Instrument/search" @{symbol = "ES"} "Instrument Search (ES)"
Test-Endpoint "POST" "/api/Instrument/search" @{symbol = "NQ"} "Instrument Search (NQ)"
Test-Endpoint "POST" "/api/Instrument/available" @{} "Instrument Available"
Test-Endpoint "GET" "/api/Instrument/ES" $null "Instrument Direct (ES)"
Test-Endpoint "GET" "/api/Instrument/NQ" $null "Instrument Direct (NQ)"

# Quote endpoints
Test-Endpoint "GET" "/api/Quote/ES" $null "Quote Direct (ES)"
Test-Endpoint "GET" "/api/Quote/NQ" $null "Quote Direct (NQ)"
Test-Endpoint "GET" "/api/Quotes/ES" $null "Quotes Direct (ES)"
Test-Endpoint "GET" "/api/MarketData/ES" $null "MarketData Direct (ES)"

# Account-related endpoints that might have contract info
Test-Endpoint "POST" "/api/Account/instruments" @{} "Account Instruments"
Test-Endpoint "POST" "/api/Account/contracts" @{} "Account Contracts"

# Position endpoints that might show contracts
Test-Endpoint "POST" "/api/Position/search" @{} "Position Search"
Test-Endpoint "GET" "/api/Positions" $null "Positions List"

# Order endpoints that might show available contracts
Test-Endpoint "POST" "/api/Order/search" @{} "Order Search"

# Try some generic endpoints
Test-Endpoint "GET" "/api/contracts" $null "Generic Contracts"
Test-Endpoint "GET" "/api/instruments" $null "Generic Instruments"
Test-Endpoint "GET" "/api/symbols" $null "Generic Symbols"

Write-Host "🎯 Discovery complete!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 If no contracts were found:" -ForegroundColor Yellow
Write-Host "1. Market might be closed (normal for eval accounts)" -ForegroundColor Gray
Write-Host "2. Try during market hours when TopstepX streams are active" -ForegroundColor Gray
Write-Host "3. Use TopstepX web app DevTools to capture contract IDs" -ForegroundColor Gray
Write-Host "4. Set environment variables with real contract IDs:" -ForegroundColor Gray
Write-Host "   Set-Item -Path 'Env:TOPSTEPX_EVAL_ES_ID' -Value '12345678'" -ForegroundColor Gray
Write-Host "   Set-Item -Path 'Env:TOPSTEPX_EVAL_NQ_ID' -Value '23456789'" -ForegroundColor Gray