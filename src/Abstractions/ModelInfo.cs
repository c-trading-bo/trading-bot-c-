using System;

namespace TradingBot.Abstractions;

/// <summary>
/// Canonical ModelInfo definition - unified across all projects
/// </summary>
public class ModelInfo
{
    public string ModelIdentifier { get; set; } = string.Empty;
    public string FullPath { get; set; } = string.Empty;
    public string FileName { get; set; } = string.Empty;
    public bool IsValidName { get; set; }
    public string ModelName { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public DateTime LastModified { get; set; }
    public DateTime DownloadedAt { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
}