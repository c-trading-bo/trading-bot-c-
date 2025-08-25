#nullable enable
using System;
using System.Net;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Health
{
    public static class HealthzServer
    {
        public static void Start(Preflight pf, string symbol, string prefix = "http://127.0.0.1:18080/", CancellationToken ct = default)
        {
            try
            {
                var listener = new HttpListener();
                if (!prefix.EndsWith("/")) prefix += "/";
                listener.Prefixes.Add(prefix);
                listener.Start();
                _ = Task.Run(async () =>
                {
                    while (!ct.IsCancellationRequested)
                    {
                        try
                        {
                            var ctx = await listener.GetContextAsync();
                            if (ctx.Request.Url != null && ctx.Request.Url.AbsolutePath.Equals("/healthz", StringComparison.OrdinalIgnoreCase))
                            {
                                var (ok, msg) = await pf.RunAsync(symbol, ct);
                                var json = JsonSerializer.Serialize(new { ok, msg });
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.ContentEncoding = System.Text.Encoding.UTF8;
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                                ctx.Response.Close();
                            }
                            else
                            {
                                ctx.Response.StatusCode = 404;
                                ctx.Response.Close();
                            }
                        }
                        catch (Exception)
                        {
                            if (ct.IsCancellationRequested) break;
                        }
                    }
                    try { listener.Stop(); listener.Close(); } catch { }
                }, ct);
            }
            catch { }
        }
    }
}
