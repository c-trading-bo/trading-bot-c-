# Simple TopstepX Contract Discovery Tool
param([string]$JwtToken = $env:TOPSTEPX_JWT)

if (-not $JwtToken) {
    Write-Host "ERROR: JWT token not found. Set TOPSTEPX_JWT environment variable" -ForegroundColor Red
    exit 1
}

Write-Host "TopstepX Contract Discovery Tool" -ForegroundColor Cyan
Write-Host "JWT Length: $($JwtToken.Length) characters" -ForegroundColor Gray
Write-Host ""

$headers = @{
    "Authorization" = "Bearer $JwtToken"
    "Content-Type" = "application/json"
}

$baseUrl = "https://api.topstepx.com"
$endpoints = @(
    @{Method="POST"; Path="/api/Contract/available"; Body='{"live": false}'; Name="Contract Available (Sim)"},
    @{Method="POST"; Path="/api/Contract/available"; Body='{"live": true}'; Name="Contract Available (Live)"},
    @{Method="POST"; Path="/api/Contract/search"; Body='{"symbol": "ES"}'; Name="Contract Search ES"},
    @{Method="POST"; Path="/api/Contract/search"; Body='{"symbol": "NQ"}'; Name="Contract Search NQ"},
    @{Method="POST"; Path="/api/Instrument/search"; Body='{"symbol": "ES"}'; Name="Instrument Search ES"},
    @{Method="POST"; Path="/api/Instrument/search"; Body='{"symbol": "NQ"}'; Name="Instrument Search NQ"},
    @{Method="GET"; Path="/api/Quote/ES"; Body=""; Name="Quote ES"},
    @{Method="GET"; Path="/api/Quote/NQ"; Body=""; Name="Quote NQ"},
    @{Method="POST"; Path="/api/Order/search"; Body='{}'; Name="Order Search"},
    @{Method="POST"; Path="/api/Position/search"; Body='{}'; Name="Position Search"}
)

foreach ($endpoint in $endpoints) {
    Write-Host "Testing: $($endpoint.Name)" -ForegroundColor Yellow
    Write-Host "  $($endpoint.Method) $($endpoint.Path)" -ForegroundColor Gray
    
    try {
        $uri = "$baseUrl$($endpoint.Path)"
        
        if ($endpoint.Method -eq "GET") {
            $response = Invoke-RestMethod -Uri $uri -Method GET -Headers $headers -TimeoutSec 10
        } else {
            $response = Invoke-RestMethod -Uri $uri -Method POST -Headers $headers -Body $endpoint.Body -TimeoutSec 10
        }
        
        Write-Host "  SUCCESS" -ForegroundColor Green
        
        # Look for contract data
        $foundContract = $false
        if ($response.data) {
            foreach ($item in $response.data) {
                if ($item.contractId -or $item.id) {
                    $contractId = if ($item.contractId) { $item.contractId } else { $item.id }
                    $symbol = if ($item.symbol) { $item.symbol } else { "Unknown" }
                    Write-Host "    Contract: $symbol -> ID: $contractId" -ForegroundColor Cyan
                    $foundContract = $true
                }
            }
        }
        
        if (-not $foundContract) {
            Write-Host "    No contract data found" -ForegroundColor DarkGray
        }
        
    } catch {
        $statusCode = if ($_.Exception.Response) { $_.Exception.Response.StatusCode } else { "Unknown" }
        Write-Host "  FAILED: $statusCode" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "Discovery complete!" -ForegroundColor Green
Write-Host ""
Write-Host "If no contracts found, try during market hours or use TopstepX web app DevTools" -ForegroundColor Yellow