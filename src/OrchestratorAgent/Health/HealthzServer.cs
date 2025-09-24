#nullable enable
using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text.Json;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using OrchestratorAgent.Infra;

namespace OrchestratorAgent.Health
{
    internal static class HealthzServer
    {
        public static void Start(Preflight pf, DstGuard dst, string symbol, string prefix = "http://127.0.0.1:18080/", CancellationToken ct = default)
            => StartWithMode(pf, dst, null, symbol, prefix, ct);

        public static void StartWithMode(Preflight pf, DstGuard dst, OrchestratorAgent.Ops.ModeController? mode, string symbol, string prefix = "http://127.0.0.1:18080/", CancellationToken ct = default, OrchestratorAgent.Ops.AppState? state = null, OrchestratorAgent.Ops.LiveLease? lease = null)
        {
            try
            {
                var listener = new HttpListener();
                if (!prefix.EndsWith('/')) prefix += "/";
                listener.Prefixes.Add(prefix);
                listener.Start();
                var startedUtc = Process.GetCurrentProcess().StartTime.ToUniversalTime();
                _ = Task.Run(async () =>
                {
                    while (!ct.IsCancellationRequested)
                    {
                        try
                        {
                            var ctx = await listener.GetContextAsync().ConfigureAwait(false);
                            var path = ctx.Request.Url?.AbsolutePath ?? string.Empty;
                            if (path.Equals("/healthz", StringComparison.OrdinalIgnoreCase))
                            {
                                var (ok, msg) = await pf.RunAsync(symbol, ct).ConfigureAwait(false);
                                var (_, warn) = dst.Check();
                                var json = JsonSerializer.Serialize(new { ok, msg, warn_dst = warn, mode = mode == null ? null : (mode.IsLive ? "LIVE" : "SHADOW") });
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.ContentEncoding = System.Text.Encoding.UTF8;
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/healthz/mode", StringComparison.OrdinalIgnoreCase))
                            {
                                var m = mode == null ? "unknown" : (mode.IsLive ? "LIVE" : "SHADOW");
                                var json = JsonSerializer.Serialize(new { mode = m, lease = lease?.HasLease ?? (bool?)null, drain = state?.DrainMode ?? (bool?)null });
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.ContentEncoding = System.Text.Encoding.UTF8;
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/build", StringComparison.OrdinalIgnoreCase))
                            {
                                var infoVer = Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                                               ?? Assembly.GetExecutingAssembly().GetName().Version?.ToString()
                                               ?? "dev";
                                var payload = new
                                {
                                    version = infoVer,
                                    pid = Environment.ProcessId,
                                    mode = mode?.IsLive == true ? "LIVE" : "SHADOW",
                                    lease = lease?.HasLease ?? (bool?)null,
                                    startedUtc
                                };
                                var json = JsonSerializer.Serialize(payload);
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/capabilities", StringComparison.OrdinalIgnoreCase))
                            {
                                var arr = Capabilities.All;
                                var json = JsonSerializer.Serialize(arr);
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/deploy/status", StringComparison.OrdinalIgnoreCase))
                            {
                                string stateDir = Path.Combine(AppContext.BaseDirectory, "state");
                                string lastPath = Path.Combine(stateDir, "last_deployed.txt");
                                string logPath = Path.Combine(stateDir, "deployments.jsonl");
                                string pending = Path.Combine(stateDir, "pending_commits.json");
                                static string readOrEmpty(string p) => File.Exists(p) ? File.ReadAllText(p) : "";
                                var json = JsonSerializer.Serialize(new
                                {
                                    lastDeployed = readOrEmpty(lastPath),
                                    historyJsonl = readOrEmpty(logPath),
                                    pendingCommits = readOrEmpty(pending)
                                });
                                var bytes = System.Text.Encoding.UTF8.GetBytes(json);
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/promote", StringComparison.OrdinalIgnoreCase) && mode != null)
                            {
                                mode.Set(OrchestratorAgent.Ops.TradeMode.Live);
                                if (state != null) state.DrainMode;
                                var bytes = System.Text.Encoding.UTF8.GetBytes("{\"ok\":true}");
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/demote", StringComparison.OrdinalIgnoreCase) && mode != null)
                            {
                                mode.Set(OrchestratorAgent.Ops.TradeMode.Shadow);
                                if (state != null) state.DrainMode = true;
                                var bytes = System.Text.Encoding.UTF8.GetBytes("{\"ok\":true}");
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
                                ctx.Response.Close();
                            }
                            else if (path.Equals("/drain", StringComparison.OrdinalIgnoreCase) && state != null)
                            {
                                state.DrainMode = true;
                                var bytes = System.Text.Encoding.UTF8.GetBytes("{\"ok\":true}");
                                ctx.Response.ContentType = "application/json";
                                ctx.Response.StatusCode = 200;
                                await ctx.Response.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
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
