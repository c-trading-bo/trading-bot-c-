using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for unified data integration service that validates historical and live data connections
/// </summary>
internal interface IUnifiedDataIntegrationService
{
    /// <summary>
    /// Validate data consistency across all sources
    /// </summary>
    Task<bool> ValidateDataConsistencyAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check historical data availability
    /// </summary>
    Task<bool> CheckHistoricalDataAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check live data connectivity
    /// </summary>
    Task<bool> CheckLiveDataAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get data integration status report
    /// </summary>
    Task<object> GetDataIntegrationStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get historical data connection status
    /// </summary>
    Task<HistoricalDataStatus> GetHistoricalDataStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get live data connection status
    /// </summary>
    Task<LiveDataStatus> GetLiveDataStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get data integration status (synchronous)
    /// </summary>
    UnifiedDataIntegrationStatus GetIntegrationStatus();
    
    /// <summary>
    /// Get recent data flow events
    /// </summary>
    List<DataFlowEvent> GetRecentDataFlowEvents(int maxCount = 50);
}

/// <summary>
/// Data integration status
/// </summary>
internal class UnifiedDataIntegrationStatus
{
    public bool IsHistoricalDataConnected { get; set; }
    public bool IsLiveDataConnected { get; set; }
    public DateTime LastHistoricalDataSync { get; set; }
    public DateTime LastLiveDataReceived { get; set; }
    public int TotalDataFlowEvents { get; set; }
    public bool IsFullyIntegrated { get; set; }
    public string StatusMessage { get; set; } = string.Empty;
}

/// <summary>
/// Data flow event for tracking data integration
/// </summary>
internal class DataFlowEvent
{
    public DateTime Timestamp { get; set; }
    public string EventType { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string Details { get; set; } = string.Empty;
    public bool Success { get; set; }
}

/// <summary>
/// Historical data connection status
/// </summary>
internal class HistoricalDataStatus
{
    public bool IsConnected { get; set; }
    public DateTime LastDataReceived { get; set; }
    public int TotalRecords { get; set; }
    public string[] DataSources { get; set; } = Array.Empty<string>();
    public string StatusMessage { get; set; } = string.Empty;
}

/// <summary>
/// Live data connection status
/// </summary>
internal class LiveDataStatus
{
    public bool IsConnected { get; set; }
    public DateTime LastDataReceived { get; set; }
    public double MessagesPerSecond { get; set; }
    public string[] DataSources { get; set; } = Array.Empty<string>();
    public string StatusMessage { get; set; } = string.Empty;
}