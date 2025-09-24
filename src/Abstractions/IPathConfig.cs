namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for file paths and directory locations
    /// Replaces hardcoded paths throughout the system
    /// </summary>
    public interface IPathConfig
    {
        /// <summary>
        /// Root directory for all bot data
        /// </summary>
        string GetDataRootPath();

        /// <summary>
        /// Directory for storing ML models
        /// </summary>
        string GetModelsPath();

        /// <summary>
        /// Directory for logs
        /// </summary>
        string GetLogsPath();

        /// <summary>
        /// Directory for state persistence
        /// </summary>
        string GetStatePath();

        /// <summary>
        /// Directory for configuration files
        /// </summary>
        string GetConfigPath();

        /// <summary>
        /// Directory for runtime work files
        /// </summary>
        string GetTempPath();

        /// <summary>
        /// Directory for backtesting results
        /// </summary>
        string GetBacktestResultsPath();

        /// <summary>
        /// Directory for trading reports
        /// </summary>
        string GetReportsPath();

        /// <summary>
        /// Path to the kill switch file
        /// </summary>
        string GetKillSwitchFilePath();

        /// <summary>
        /// Directory for ONNX runtime binaries
        /// </summary>
        string GetOnnxRuntimePath();

        /// <summary>
        /// Directory for Python script execution
        /// </summary>
        string GetPythonScriptsPath();
    }
}