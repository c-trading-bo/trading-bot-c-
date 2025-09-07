using System;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace BotCore.ML
{
    public class UCBManager
    {
        private readonly HttpClient _http;
        private readonly ILogger<UCBManager> _logger;
        private static readonly JsonSerializerSettings JsonCfg = new()
        {
            ContractResolver = new DefaultContractResolver { NamingStrategy = new DefaultNamingStrategy() },
            MissingMemberHandling = MissingMemberHandling.Ignore
        };

        public UCBManager(HttpClient http, ILogger<UCBManager> logger)
        {
            _logger = logger;
            _http = http;
        }

        // Helper to generate request IDs
        private string GenerateRequestId() => Guid.NewGuid().ToString("N").Substring(0, 8);

        // Helper to create JSON POST with X-Req-Id
        private HttpRequestMessage JsonPost(string path, object payload, string reqId)
        {
            var msg = new HttpRequestMessage(HttpMethod.Post, path)
            {
                Content = new StringContent(
                    JsonConvert.SerializeObject(payload, JsonCfg), 
                    Encoding.UTF8, 
                    "application/json"
                )
            };
            msg.Headers.Add("X-Req-Id", reqId);
            return msg;
        }

        public async Task<UCBRecommendation> GetRecommendationAsync(MarketData data, CancellationToken ct = default)
        {
            var requestId = GenerateRequestId();
            
            var marketJson = new
            {
                es_price = data.ESPrice,
                nq_price = data.NQPrice,
                es_volume = data.ESVolume,
                nq_volume = data.NQVolume,
                es_atr = data.ES_ATR,
                nq_atr = data.NQ_ATR,
                vix = data.VIX,
                tick = data.TICK,
                add = data.ADD,
                correlation = data.Correlation,
                rsi_es = data.RSI_ES,
                rsi_nq = data.RSI_NQ,
                instrument = data.PrimaryInstrument
            };

            using var request = JsonPost("ucb/recommend", marketJson, requestId);
            using var resp = await _http.SendAsync(request, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            var text = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

            var rec = JsonConvert.DeserializeObject<UCBRecommendation>(text, JsonCfg);
            if (rec == null) throw new InvalidOperationException("Null UCBRecommendation");
            
            _logger.LogInformation($"[{requestId}] UCB: {rec.Strategy} | Confidence: {rec.Confidence:P} | Size: {rec.PositionSize}");
            return rec;
        }

        public async Task UpdatePnLAsync(string strategy, decimal pnl, CancellationToken ct = default)
        {
            var reqId = GenerateRequestId();
            using var req = JsonPost("ucb/update_pnl", new { strategy, pnl }, reqId);
            using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            _logger.LogInformation($"[{reqId}] Updated PnL for {strategy}: ${pnl:F2}");
        }

        public async Task ResetDailyAsync(CancellationToken ct = default)
        {
            var reqId = GenerateRequestId();
            using var req = JsonPost("ucb/reset_daily", new { }, reqId);
            using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            _logger.LogInformation($"[{reqId}] UCB daily stats reset");
        }

        public async Task<TopStepLimits> CheckLimits(CancellationToken ct = default)
        {
            using var resp = await _http.GetAsync("ucb/limits", ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            var text = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            
            var dto = JsonConvert.DeserializeObject<LimitsResponse>(text);
            return new TopStepLimits
            {
                CanTrade = dto.CanTrade,
                Reason = dto.Reason ?? "OK",
                CurrentDrawdown = (decimal)(dto.CurrentDrawdown ?? 0),
                DailyPnL = (decimal)(dto.DailyPnl ?? 0)
            };
        }

        public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
        {
            try
            {
                using var resp = await _http.GetAsync("health", ct).ConfigureAwait(false);
                return resp.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        private class LimitsResponse
        {
            [JsonProperty("can_trade")] public bool CanTrade { get; set; }
            [JsonProperty("reason")] public string Reason { get; set; }
            [JsonProperty("warning")] public string Warning { get; set; }
            [JsonProperty("current_drawdown")] public double? CurrentDrawdown { get; set; }
            [JsonProperty("daily_pnl")] public double? DailyPnl { get; set; }
            [JsonProperty("account_balance")] public double? AccountBalance { get; set; }
        }
    }

    public class UCBRecommendation
    {
        public string Strategy { get; set; } = string.Empty;
        public bool Trade { get; set; }
        public decimal PositionSize { get; set; }
        public decimal Confidence { get; set; }
        public string Reasoning { get; set; } = string.Empty;
    }

    public class UCBRecommendationResponse
    {
        public string Strategy { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public bool Trade { get; set; }
        public string Reasoning { get; set; } = string.Empty;
        public string RiskLevel { get; set; } = string.Empty;
        public string Timestamp { get; set; } = string.Empty;
    }

    public class MarketData
    {
        public decimal ESPrice { get; set; }
        public decimal NQPrice { get; set; }
        public long ESVolume { get; set; }
        public long NQVolume { get; set; }
        public decimal ES_ATR { get; set; }
        public decimal NQ_ATR { get; set; }
        public decimal VIX { get; set; }
        public int TICK { get; set; }
        public int ADD { get; set; }
        public decimal Correlation { get; set; }
        public decimal RSI_ES { get; set; }
        public decimal RSI_NQ { get; set; }
        public string PrimaryInstrument { get; set; } = "ES";
    }

    public class TopStepLimits
    {
        public bool CanTrade { get; set; }
        public string Reason { get; set; } = string.Empty;
        public decimal CurrentDrawdown { get; set; }
        public decimal DailyPnL { get; set; }
    }
}
