using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Trading.Safety;

namespace OrchestratorAgent
{
    internal static class OrderSmokeTester
    {
        public static async Task RunAsync(HttpClient http, Func<Task<string?>> tokenProvider, long accountId, string contractId, ILogger log, CancellationToken ct)
        {
            try
            {
                var token = await tokenProvider().ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(token))
                {
                    log.LogWarning("[SmokeTest] Skipping: missing JWT token.");
                    return;
                }

                // 1) Place a 1-lot LIMIT far away (won’t fill). Using an extremely low price for a BUY.
                var placeBody = new
                {
                    accountId,
                    contractId,
                    type = 1,   // 1 = Limit
                    side = 0,   // 0 = Buy, 1 = Sell
                    size = 1,
                    limitPrice = 1000m,
                    customTag = $"smoketest-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}"
                };

                using var placeReq = new HttpRequestMessage(HttpMethod.Post, "/api/Order/place");
                placeReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                placeReq.Content = new StringContent(JsonSerializer.Serialize(placeBody), Encoding.UTF8, "application/json");

                log.LogInformation("[SmokeTest] Placing far-away LIMIT order: {Contract} acct={Account}", contractId, SecurityHelpers.MaskAccountId(accountId));
                using var placeResp = await http.SendAsync(placeReq, ct).ConfigureAwait(false);
                var placeText = await placeResp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                if (!placeResp.IsSuccessStatusCode)
                {
                    log.LogWarning("[SmokeTest] Place failed {Status}: {Body}", (int)placeResp.StatusCode, Trunc(placeText));
                    return;
                }

                long orderId = 0;
                try
                {
                    using var doc = JsonDocument.Parse(placeText);
                    if (doc.RootElement.TryGetProperty("orderId", out var oid)) orderId = oid.GetInt64();
                    else if (doc.RootElement.TryGetProperty("id", out var idEl)) orderId = idEl.GetInt64();
                }
                catch { }

                if (orderId <= 0)
                {
                    log.LogWarning("[SmokeTest] Could not parse orderId from response: {Body}", Trunc(placeText));
                    return;
                }

                // 2) Cancel the order
                var cancelBody = new { orderId };
                using var cancelReq = new HttpRequestMessage(HttpMethod.Post, "/api/Order/cancel");
                cancelReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                cancelReq.Content = new StringContent(JsonSerializer.Serialize(cancelBody), Encoding.UTF8, "application/json");

                log.LogInformation("[SmokeTest] Canceling orderId={OrderId}", SecurityHelpers.MaskOrderId(orderId.ToString()));
                using var cancelResp = await http.SendAsync(cancelReq, ct).ConfigureAwait(false);
                var cancelText = await cancelResp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                if (!cancelResp.IsSuccessStatusCode)
                {
                    log.LogWarning("[SmokeTest] Cancel failed {Status}: {Body}", (int)cancelResp.StatusCode, Trunc(cancelText));
                    return;
                }

                log.LogInformation("[SmokeTest] Place+Cancel completed successfully for orderId={OrderId}", SecurityHelpers.MaskOrderId(orderId.ToString()));
            }
            catch (OperationCanceledException)
            {
                // ignore
            }
            catch (Exception ex)
            {
                log.LogWarning(ex, "[SmokeTest] Unexpected error during place/cancel test.");
            }
        }

        private static string Trunc(string? s, int max = 512)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            return s.Length <= max ? s : s[..max] + "…";
        }
    }
}
