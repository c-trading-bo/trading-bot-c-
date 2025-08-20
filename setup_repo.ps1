# === TopstepX repo cleanup & upgrade ===
# Run from the folder that contains: "C# ai bot.sln" and projects like BotCore/, PlaceOrderTest/, etc.

$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
Write-Host "Repo root: $root"

# --- Create new layout ---
$src = Join-Path $root 'src'
$examples = Join-Path $root 'examples'
$tests = Join-Path $root 'tests'
New-Item -ItemType Directory -Force -Path $src, $examples, $tests | Out-Null

# --- Move projects into the new layout (if they exist) ---
if (Test-Path "$root\BotCore")        { Move-Item "$root\BotCore"        "$src\BotCore"        -Force }
if (Test-Path "$root\StrategyAgent")  { Move-Item "$root\StrategyAgent"  "$src\StrategyAgent"  -Force }
if (Test-Path "$root\PlaceOrderTest") { Move-Item "$root\PlaceOrderTest" "$examples\PlaceOrderTest" -Force }
if (Test-Path "$root\BotTests")       { Move-Item "$root\BotTests"       "$tests\BotTests"     -Force }

# --- Quarantine the old top-level runner so it stops conflicting ---
$newLegacy = Join-Path $examples 'LegacyRunner'
New-Item -ItemType Directory -Path $newLegacy -Force | Out-Null
if (Test-Path "$root\C# ai bot.csproj") { Move-Item "$root\C# ai bot.csproj" "$newLegacy\LegacyRunner.csproj" -Force }
if (Test-Path "$root\Program.cs")       { Move-Item "$root\Program.cs"       "$newLegacy\Program.cs"         -Force }
if (Test-Path "$root\PlaceOrderAndWatchFill.cs") { Move-Item "$root\PlaceOrderAndWatchFill.cs" "$newLegacy\PlaceOrderAndWatchFill.cs" -Force }

# --- Fix project references for the new layout ---
$placeOrderProj = "$examples\PlaceOrderTest\PlaceOrderTest.csproj"
if (Test-Path $placeOrderProj) {
  (Get-Content $placeOrderProj) `
    -replace '\.\.\\BotCore\\BotCore.csproj','..\\..\\src\\BotCore\\BotCore.csproj' `
    -replace '\.\.\\StrategyAgent\\StrategyAgent.csproj','..\\..\\src\\StrategyAgent\\StrategyAgent.csproj' |
    Set-Content $placeOrderProj -Encoding UTF8
}
$strategyProj = "$src\StrategyAgent\StrategyAgent.csproj"
if (Test-Path $strategyProj) {
  (Get-Content $strategyProj) `
    -replace '\.\.\\BotCore\\BotCore.csproj','..\\src\\BotCore\\BotCore.csproj' |
    Set-Content $strategyProj -Encoding UTF8
}

# --- .gitignore & env hygiene ---
$gi = Join-Path $root ".gitignore"
if (-not (Test-Path $gi)) {
@"
bin/
obj/
**/*.user
**/*.suo
.vscode/
.env.local
"@ | Set-Content $gi -Encoding UTF8
} else {
  if (-not (Select-String -Path $gi -Pattern '^\Q.env.local\E$' -SimpleMatch -Quiet)) {
    Add-Content $gi "`n.env.local"
  }
}

# Make sure .env.example has placeholders (do NOT leak secrets)
$envExample = Join-Path $root ".env.example"
if (Test-Path $envExample) {
  (Get-Content $envExample) `
    -replace '^TSX_USERNAME=.*','TSX_USERNAME=<your email>' `
    -replace '^TSX_API_KEY=.*','TSX_API_KEY=<your api key>' `
    -replace '^PROJECTX_USERNAME=.*','PROJECTX_USERNAME=<your email>' `
    -replace '^PROJECTX_APIKEY=.*','PROJECTX_APIKEY=<your api key>' `
    -replace '^TOPSTEPX_JWT=.*','TOPSTEPX_JWT=<runtime token from auth>' `
    | Set-Content $envExample -Encoding UTF8
}

# --- Purge build artifacts ---
Get-ChildItem $root -Recurse -Force -Include bin,obj | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue

# --- Add a resilient SignalR agent (non-destructive new file) ---
$agentDir = Join-Path $src 'BotCore'
$newAgent = Join-Path $agentDir 'ReliableMarketDataAgent.cs'
$agentCode = @'
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using BotCore.Models;

namespace BotCore
{
    /// <summary>
    /// Resilient SignalR market data client:
    ///  - awaits StartAsync before sending
    ///  - gates sends on Connected state
    ///  - resubscribes after automatic reconnect
    ///  - wires both "Gateway*" and plain event names
    /// </summary>
    public sealed class ReliableMarketDataAgent : IAsyncDisposable
    {
        private HubConnection? _hub;
        private readonly string _jwt;
        private string? _contractId;
        private string? _barTf;

        public event Action<Bar>? OnBar;
        public event Action<JsonElement>? OnQuote;
        public event Action<JsonElement>? OnTrade;

        public ReliableMarketDataAgent(string jwt)
        {
            _jwt = jwt ?? throw new ArgumentNullException(nameof(jwt));
        }

        public async Task StartAsync(string contractId, string barTf, CancellationToken ct = default)
        {
            _contractId = contractId ?? throw new ArgumentNullException(nameof(contractId));
            _barTf = barTf ?? throw new ArgumentNullException(nameof(barTf));

            _hub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", o =>
                {
                    o.AccessTokenProvider = () => Task.FromResult(_jwt);
                    o.SkipNegotiation = true;
                    o.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                })
                .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(30) })
                .Build();

            WireHandlers(_hub);

            _hub.Reconnected += async _ =>
            {
                try { await SubscribeAll(ct).ConfigureAwait(false); }
                catch (Exception ex) { Console.WriteLine($"[ReliableMarketDataAgent] Resubscribe failed: {ex.Message}"); }
            };

            await _hub.StartAsync(ct).ConfigureAwait(false);
            await WaitForConnectedAsync(ct).ConfigureAwait(false);
            await SubscribeAll(ct).ConfigureAwait(false);
        }

        private void WireHandlers(HubConnection hub)
        {
            hub.On<JsonElement>("Bar", data =>
            {
                try
                {
                    var b = new Bar
                    {
                        Ts = data.TryGetProperty("ts", out var ts) ? ts.GetInt64() :
                             data.TryGetProperty("timestamp", out var t2) ? t2.GetInt64() : 0,
                        Open = data.TryGetProperty("open", out var o) ? o.GetDecimal() : 0m,
                        High = data.TryGetProperty("high", out var h) ? h.GetDecimal() : 0m,
                        Low = data.TryGetProperty("low", out var l) ? l.GetDecimal() : 0m,
                        Close = data.TryGetProperty("close", out var c) ? c.GetDecimal() : 0m,
                        Volume = data.TryGetProperty("volume", out var v) ? v.GetInt32() : 0,
                        Symbol = data.TryGetProperty("symbol", out var s) ? s.GetString() ?? "" : ""
                    };
                    OnBar?.Invoke(b);
                }
                catch { }
            });
            hub.On<JsonElement>("GatewayBars", data => OnQuote?.Invoke(data)); // fallback

            hub.On<JsonElement>("Quote", data => OnQuote?.Invoke(data));
            hub.On<JsonElement>("GatewayQuote", data => OnQuote?.Invoke(data));

            hub.On<JsonElement>("Trade", data => OnTrade?.Invoke(data));
            hub.On<JsonElement>("GatewayTrade", data => OnTrade?.Invoke(data));
        }

        private async Task SubscribeAll(CancellationToken ct)
        {
            if (_hub is null) throw new InvalidOperationException("Hub is not built.");
            if (_contractId is null || _barTf is null) throw new InvalidOperationException("Call StartAsync(contractId, barTf) first.");
            await WaitForConnectedAsync(ct).ConfigureAwait(false);

            await TrySendAsync("SubscribeQuote", new object?[] { _contractId }, ct);
            await TrySendAsync("SubscribeContractQuotes", new object?[] { _contractId }, ct);

            await TrySendAsync("SubscribeTrade", new object?[] { _contractId }, ct);
            await TrySendAsync("SubscribeContractTrades", new object?[] { _contractId }, ct);

            await TrySendAsync("SubscribeBars", new object?[] { _contractId, _barTf }, ct);
            await TrySendAsync("SubscribeContractBars", new object?[] { _contractId, _barTf }, ct);

            Console.WriteLine($"[ReliableMarketDataAgent] Subscribed to {_contractId} ({_barTf}).");
        }

        private async Task<bool> TrySendAsync(string method, object?[] args, CancellationToken ct)
        {
            try
            {
                if (_hub is null) return false;
                if (_hub.State != HubConnectionState.Connected) await WaitForConnectedAsync(ct).ConfigureAwait(false);
                await _hub.SendAsync(method, args, ct).ConfigureAwait(false);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ReliableMarketDataAgent] {method} failed: {ex.Message}");
                return false;
            }
        }

        private async Task WaitForConnectedAsync(CancellationToken ct)
        {
            if (_hub is null) throw new InvalidOperationException("Hub is null");
            var start = DateTime.UtcNow;
            while (_hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if ((DateTime.UtcNow - start) > TimeSpan.FromSeconds(15))
                    throw new TimeoutException("SignalR connection did not reach Connected state within 15s.");
                await Task.Delay(200, ct).ConfigureAwait(false);
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_hub is not null)
            {
                try { await _hub.DisposeAsync(); } catch { }
            }
        }

        public async Task StopAsync(CancellationToken ct = default)
        {
            if (_hub is not null)
            {
                try { await _hub.StopAsync(ct).ConfigureAwait(false); } catch { }
                await DisposeAsync();
            }
        }
    }
}
'@
Set-Content $newAgent $agentCode -Encoding UTF8
Write-Host "Added BotCore\ReliableMarketDataAgent.cs"
$exampleDir = "$examples\PlaceOrderTest"
if (Test-Path $exampleDir) {
  Get-ChildItem $exampleDir -Recurse -Filter *.cs | ForEach-Object {
    $p = $_.FullName
    $txt = Get-Content $p -Raw
    $newTxt = $txt -replace '\bMarketDataAgent\b','ReliableMarketDataAgent'
    if ($newTxt -ne $txt) {
      Set-Content $p $newTxt -Encoding UTF8
      Write-Host "Updated: $($_.Name) to use ReliableMarketDataAgent"
    }
  }
}
$oldSln = Join-Path $root "C# ai bot.sln"
$newSln = Join-Path $root "TopstepX.Bot.sln"
if (Test-Path $newSln) { Remove-Item $newSln -Force }
dotnet new sln -n "TopstepX.Bot" | Out-Null
if (Test-Path "$src\BotCore\BotCore.csproj")              { dotnet sln $newSln add "$src\BotCore\BotCore.csproj"              | Out-Null }
if (Test-Path "$src\StrategyAgent\StrategyAgent.csproj")  { dotnet sln $newSln add "$src\StrategyAgent\StrategyAgent.csproj"  | Out-Null }
if (Test-Path "$examples\PlaceOrderTest\PlaceOrderTest.csproj") { dotnet sln $newSln add "$examples\PlaceOrderTest\PlaceOrderTest.csproj" | Out-Null }
if (Test-Path "$tests\BotTests\BotTests.csproj")          { dotnet sln $newSln add "$tests\BotTests\BotTests.csproj"          | Out-Null }
dotnet restore
dotnet build --nologo
Write-Host "`nDone. Open TopstepX.Bot.sln and use the example project under /examples."
