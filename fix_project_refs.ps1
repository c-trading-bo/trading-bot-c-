$ErrorActionPreference = "Stop"
$root = "C:\Users\kevin\Downloads\C# ai bot"
Set-Location $root

# 1) Ensure src\BotCore exists and (re)create BotCore.csproj if missing
$botCoreDir = Join-Path $root "src\BotCore"
if (-not (Test-Path $botCoreDir)) {
  throw "Folder not found: $botCoreDir. Create 'src\BotCore' and move your BotCore *.cs files there first."
}
$botCoreProj = Join-Path $botCoreDir "BotCore.csproj"
if (-not (Test-Path $botCoreProj)) {
@'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.AspNetCore.SignalR.Client" Version="9.0.8" />
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>
</Project>
'@ | Set-Content $botCoreProj -Encoding UTF8
  Write-Host "Created: src\BotCore\BotCore.csproj"
} else {
  Write-Host "Found: src\BotCore\BotCore.csproj"
}

# 2) Fix StrategyAgent.csproj to reference ..\BotCore\BotCore.csproj (src siblings)
$strategyProj = Join-Path $root "src\StrategyAgent\StrategyAgent.csproj"
if (Test-Path $strategyProj) {
  $txt = Get-Content $strategyProj -Raw
  $new = $txt -replace '<ProjectReference Include="[^"]*BotCore\.csproj"\s*/>',
                    '<ProjectReference Include="..\BotCore\BotCore.csproj" />'
  if ($new -ne $txt) {
    $new | Set-Content $strategyProj -Encoding UTF8
    Write-Host "Updated StrategyAgent.csproj project reference."
  } else {
    Write-Host "StrategyAgent.csproj reference looks correct."
  }
} else {
  Write-Host "WARNING: Missing file src\StrategyAgent\StrategyAgent.csproj"
}

# 3) Fix PlaceOrderTest.csproj project refs to point up two levels into src\
$placeOrderProj = Join-Path $root "examples\PlaceOrderTest\PlaceOrderTest.csproj"
if (Test-Path $placeOrderProj) {
  $txt = Get-Content $placeOrderProj -Raw
  $new = $txt
  # BotCore ref: ..\..\src\BotCore\BotCore.csproj
  $new = $new -replace '<ProjectReference Include="[^"]*BotCore\.csproj"\s*/>',
                    '<ProjectReference Include="..\..\src\BotCore\BotCore.csproj" />'
  # StrategyAgent ref: ..\..\src\StrategyAgent\StrategyAgent.csproj
  $new = $new -replace '<ProjectReference Include="[^"]*StrategyAgent\.csproj"\s*/>',
                    '<ProjectReference Include="..\..\src\StrategyAgent\StrategyAgent.csproj" />'
  if ($new -ne $txt) {
    $new | Set-Content $placeOrderProj -Encoding UTF8
    Write-Host "Updated PlaceOrderTest.csproj project references."
  } else {
    Write-Host "PlaceOrderTest.csproj references already good."
  }
} else {
  Write-Host "WARNING: Missing file examples\PlaceOrderTest\PlaceOrderTest.csproj"
}

# 4) Optional: ensure a solution exists and includes all projects (safe to re-create)
$sln = Join-Path $root "TopstepX.Bot.sln"
if (Test-Path $sln) { Remove-Item $sln -Force }
dotnet new sln -n "TopstepX.Bot" | Out-Null
if (Test-Path $botCoreProj) { dotnet sln $sln add $botCoreProj | Out-Null }
if (Test-Path $strategyProj) { dotnet sln $sln add $strategyProj | Out-Null }
if (Test-Path $placeOrderProj) { dotnet sln $sln add $placeOrderProj | Out-Null }

# 5) Restore/build and show what we fixed
Write-Host "`n=== dotnet restore ==="
dotnet restore

Write-Host "`n=== dotnet build (Release) ==="
dotnet build -c Release --nologo

Write-Host "`nAll set. Run the example:"
Write-Host 'dotnet run --project .\examples\PlaceOrderTest\PlaceOrderTest.csproj'
