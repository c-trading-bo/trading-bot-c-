#nullable enable
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;

namespace BotCore
{
    // Minimal history client using ApiClient
    public sealed class HistoryApi(ApiClient api)
    {
        private readonly ApiClient _api = api;

        public async Task<List<Bar>> GetMinuteBarsAsync(string symbol, long sinceUnixMs, long untilUnixMs, CancellationToken ct = default)
        {
            var path = $"/marketdata/bars?symbol={symbol}&tf=1m&since={sinceUnixMs}&until={untilUnixMs}";
            var list = await _api.GetAsync<List<Bar>>(path, ct).ConfigureAwait(false);
            return list ?? [];
        }
    }
}
