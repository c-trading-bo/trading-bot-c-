using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;

namespace OrchestratorAgent.Execution
{
    public sealed class FullAutoScheduler : BackgroundService
    {
        readonly ILogger<FullAutoScheduler> _log;
        readonly IServiceProvider _sp;
        readonly TimeSpan _at;
        readonly int _lookbackDays, _testDays;

        public FullAutoScheduler(ILogger<FullAutoScheduler> log, IServiceProvider sp)
        {
            _log = log;
            _sp = sp;
            _at = TimeSpan.Parse(Environment.GetEnvironmentVariable("AUTO_RETUNE_UTC") ?? "02:45");
            _lookbackDays = int.TryParse(Environment.GetEnvironmentVariable("AUTO_RETUNE_LOOKBACK_DAYS"), out var lb) ? lb : 30;
            _testDays = int.TryParse(Environment.GetEnvironmentVariable("AUTO_RETUNE_TEST_DAYS"), out var td) ? td : 5;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!Flag("AUTO_RETUNE_ENABLED")) 
            {
                _log.LogInformation("[AUTO] Auto-retune disabled (AUTO_RETUNE_ENABLED != 1)");
                return;
            }

            _log.LogInformation("[AUTO] Auto-retune scheduler started at UTC {Time}", _at);

            while (!stoppingToken.IsCancellationRequested)
            {
                var now = DateTime.UtcNow.TimeOfDay;
                var delay = now <= _at ? _at - now : TimeSpan.FromDays(1) - (now - _at);
                
                try
                {
                    await Task.Delay(delay, stoppingToken);
                    await RetuneAsync(stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    try { _log.LogError(ex, "[AUTO] retune failed"); }
                    catch { }
                }
            }
        }

        async Task RetuneAsync(CancellationToken ct)
        {
            _log.LogInformation("[AUTO] nightly retune start lookback={LB} test={TD}", _lookbackDays, _testDays);
            
            try
            {
                // 1) Run walk-forward backtests over last LB days, folds of TEST days
                // 2) Compute priors per (strat,cfg,regime,session) and write strat-configs + priors JSON
                // 3) Signal orchestrator to reload lattice/priors (in-proc event or file watcher)
                
                // For now, just log that we would retune
                _log.LogInformation("[AUTO] Would run walk-forward validation on {Days} days with {TestDays}-day folds", _lookbackDays, _testDays);
                
                // TODO: Integrate with existing TuningRunner
                // var tuningRunner = _sp.GetService<TuningRunner>();
                // await tuningRunner.RunWalkForwardAsync(_lookbackDays, _testDays, ct);
                
                await Task.CompletedTask;
                _log.LogInformation("[AUTO] nightly retune done");
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AUTO] retune execution failed");
                throw;
            }
        }

        static bool Flag(string k)
        {
            var v = Environment.GetEnvironmentVariable(k)?.ToLowerInvariant();
            return v == "1" || v == "true";
        }
    }
}
