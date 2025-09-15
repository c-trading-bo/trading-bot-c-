using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Security;

/// <summary>
/// Interface for immutable audit logging with tamper detection
/// </summary>
public interface IAuditImmutabilityManager
{
    /// <summary>
    /// Initialize audit storage with immutability configuration
    /// </summary>
    Task InitializeAsync(AuditImmutabilityConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Write audit log entry with immutability protection
    /// </summary>
    Task<string> WriteAuditEntryAsync(AuditLogEntry entry, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Verify integrity of audit log entries
    /// </summary>
    Task<AuditIntegrityReport> VerifyAuditIntegrityAsync(AuditIntegrityCheck check, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search audit entries with integrity validation
    /// </summary>
    Task<List<AuditLogEntry>> SearchAuditEntriesAsync(AuditSearchCriteria criteria, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Archive old audit entries to long-term storage
    /// </summary>
    Task<AuditArchiveResult> ArchiveAuditEntriesAsync(DateTime cutoffDate, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get audit storage statistics and health
    /// </summary>
    Task<AuditStorageMetrics> GetStorageMetricsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for audit immutability
/// </summary>
public record AuditImmutabilityConfiguration(
    string StoragePath,
    AuditStorageFormat Format,
    bool EnableEncryption = true,
    string? EncryptionKey = null,
    TimeSpan RetentionPeriod = default,
    bool EnableCompression = true,
    int MaxEntriesPerFile = 10000
)
{
    public TimeSpan RetentionPeriod { get; init; } = RetentionPeriod == default ? TimeSpan.FromDays(2555) : RetentionPeriod; // 7 years
}

/// <summary>
/// Storage formats for audit logs
/// </summary>
public enum AuditStorageFormat
{
    JsonLines,
    BinaryProtobuf,
    ParquetColumnar
}

/// <summary>
/// Immutable audit log entry
/// </summary>
public record AuditLogEntry(
    string EntryId,
    DateTime Timestamp,
    string UserId,
    string Action,
    string Resource,
    Dictionary<string, object> Details,
    string Hash,
    string? PreviousEntryHash = null,
    string? Signature = null
);

/// <summary>
/// Criteria for searching audit entries
/// </summary>
public record AuditSearchCriteria(
    DateTime? StartTime = null,
    DateTime? EndTime = null,
    string? UserId = null,
    string? Action = null,
    string? Resource = null,
    Dictionary<string, object>? DetailFilters = null,
    int MaxResults = 1000
);

/// <summary>
/// Configuration for audit integrity checks
/// </summary>
public record AuditIntegrityCheck(
    DateTime? StartTime = null,
    DateTime? EndTime = null,
    bool CheckChainIntegrity = true,
    bool CheckSignatures = true,
    bool CheckFileIntegrity = true
);

/// <summary>
/// Report of audit integrity verification
/// </summary>
public record AuditIntegrityReport(
    DateTime GeneratedAt,
    bool IntegrityValid,
    int TotalEntriesChecked,
    int CorruptedEntries,
    List<AuditIntegrityIssue> Issues,
    Dictionary<string, object> Statistics
);

/// <summary>
/// Individual audit integrity issue
/// </summary>
public record AuditIntegrityIssue(
    AuditIntegrityIssueType Type,
    string EntryId,
    DateTime EntryTimestamp,
    string Description,
    AuditIntegrityIssueSeverity Severity
);

/// <summary>
/// Types of audit integrity issues
/// </summary>
public enum AuditIntegrityIssueType
{
    HashMismatch,
    ChainBroken,
    SignatureInvalid,
    EntryCorrupted,
    FileMissing,
    TimestampOutOfOrder
}

/// <summary>
/// Severity of audit integrity issues
/// </summary>
public enum AuditIntegrityIssueSeverity
{
    Warning,
    Error,
    Critical
}

/// <summary>
/// Result of audit archiving operation
/// </summary>
public record AuditArchiveResult(
    bool Success,
    int EntriesArchived,
    long BytesArchived,
    string ArchivePath,
    DateTime ArchivedAt,
    string? Error = null
);

/// <summary>
/// Metrics for audit storage system
/// </summary>
public record AuditStorageMetrics(
    DateTime GeneratedAt,
    long TotalEntries,
    long TotalSizeBytes,
    DateTime OldestEntry,
    DateTime NewestEntry,
    double AverageEntrySizeBytes,
    int FilesCount,
    bool IntegrityHealthy
);