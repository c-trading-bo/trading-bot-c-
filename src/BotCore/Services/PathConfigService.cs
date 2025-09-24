using System;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of path configuration
    /// Replaces hardcoded file paths and directory locations
    /// </summary>
    public class PathConfigService : IPathConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<PathConfigService> _logger;

        public PathConfigService(IConfiguration config, ILogger<PathConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public string GetDataRootPath() => 
            _config.GetValue("Paths:DataRoot", Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TradingBot"));

        public string GetModelsPath() => 
            _config.GetValue("Paths:Models", Path.Combine(GetDataRootPath(), "models"));

        public string GetLogsPath() => 
            _config.GetValue("Paths:Logs", Path.Combine(GetDataRootPath(), "logs"));

        public string GetStatePath() => 
            _config.GetValue("Paths:State", Path.Combine(GetDataRootPath(), "state"));

        public string GetConfigPath() => 
            _config.GetValue("Paths:Config", Path.Combine(GetDataRootPath(), "config"));

        public string GetTempPath() => 
            _config.GetValue("Paths:Temp", Path.Combine(Path.GetTempPath(), "TradingBot"));

        public string GetBacktestResultsPath() => 
            _config.GetValue("Paths:BacktestResults", Path.Combine(GetDataRootPath(), "backtests"));

        public string GetReportsPath() => 
            _config.GetValue("Paths:Reports", Path.Combine(GetDataRootPath(), "reports"));

        public string GetKillSwitchFilePath() => 
            _config.GetValue("Paths:KillSwitchFile", Path.Combine(GetDataRootPath(), "kill.txt"));

        public string GetOnnxRuntimePath() => 
            _config.GetValue("Paths:OnnxRuntime", Path.Combine(GetDataRootPath(), "onnx"));

        public string GetPythonScriptsPath() => 
            _config.GetValue("Paths:PythonScripts", Path.Combine(GetDataRootPath(), "scripts"));
    }
}