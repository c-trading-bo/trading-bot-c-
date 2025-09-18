using System;

namespace TradingBot.Abstractions;

/// <summary>
/// Decision service status information
/// </summary>
public class DecisionServiceStatus
{
    public bool PythonServiceHealthy { get; set; }
    public DateTime LastHealthCheck { get; set; }
    public string ServiceEndpoint { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public int TimeoutMs { get; set; }
    public int MaxRetries { get; set; }
}

/// <summary>
/// Decision service configuration options
/// </summary>
public class DecisionServiceOptions
{
    public Uri? BaseUrl { get; set; }
    public int TimeoutMs { get; set; } = 5000;
    public bool Enabled { get; set; } = true;
    public int MaxRetries { get; set; } = 3;
}