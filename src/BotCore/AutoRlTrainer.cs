using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using BotCore.Utilities;

namespace BotCore
{
    /// <summary>
    /// DEPRECATED: Local automated RL trainer - replaced by 100% cloud-based learning.
    /// This component is no longer used as all training now happens in the cloud via GitHub Actions.
    /// The bot only needs to run for trading, not for learning.
    /// Use CloudDataUploader + CloudRlTrainerEnhanced for cloud-based learning instead.
    /// </summary>
    [Obsolete("Local training is deprecated. Use 100% cloud-based learning instead.", false)]
    public sealed class AutoRlTrainer : IDisposable
    {
        private readonly ILogger _log;
        private readonly Timer? _timer;
        private readonly string? _dataDir;
        private readonly string? _modelDir;
        private readonly string? _pythonScriptDir;
        private bool _disposed;

        public AutoRlTrainer(ILogger logger)
        {
            _log = logger;

            // Log deprecation warning using standardized helper
            LoggingHelper.LogDeprecation(_log, "AutoRlTrainer", "Local training is disabled in favor of 100% cloud-based learning", "CloudDataUploader + CloudRlTrainerEnhanced");

            // Don't start the timer - this component is deprecated
            _log.LogInformation("[AutoRlTrainer] Local training disabled - all learning now happens in cloud every 30 minutes");

            // Initialize as null since this is deprecated
            _timer = null;
            _dataDir = null;
            _modelDir = null;
            _pythonScriptDir = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _timer?.Dispose();
            _log.LogInformation("[AutoRlTrainer] Stopped");
        }
    }
}
