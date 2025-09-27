using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Suppression ledger service for tracking and justifying analyzer suppressions
    /// Maintains a central record of all suppressions with justifications
    /// Uses LoggerMessage delegates for improved performance (CA1848 compliance)
    /// </summary>
    public class SuppressionLedgerService
    {
        // Constants for analyzer suppression processing
        private const int MaxLineNumberDrift = 2;           // Tolerance for line number changes in suppressions
        private const int PragmaRuleIdIndex = 3;            // Index of rule ID in pragma warning disable statements
        
        private readonly ILogger<SuppressionLedgerService> _logger;
        private readonly string _ledgerPath;
        private readonly List<SuppressionEntry> _suppressions = new();
        private readonly object _ledgerLock = new();

        // Cached JsonSerializerOptions for performance (CA1869 compliance)
        private static readonly JsonSerializerOptions JsonOptions = new()
        { 
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };

        // LoggerMessage delegates for improved performance (CA1848 compliance)
        private static readonly Action<ILogger, int, Exception?> _logLedgerInitialized = 
            LoggerMessage.Define<int>(LogLevel.Information, new EventId(2001, "LedgerInit"), 
                "📋 [SUPPRESSION] Suppression ledger initialized with {Count} entries");
                
        private static readonly Action<ILogger, string, string, int, string, string, Exception?> _logSuppressionRecorded = 
            LoggerMessage.Define<string, string, int, string, string>(LogLevel.Warning, new EventId(2002, "SuppressionRecorded"), 
                "⚠️ [SUPPRESSION] Recorded suppression {RuleId} in {File}:{Line} by {Author}: {Justification}");
                
        private static readonly Action<ILogger, Guid, string, string, Exception?> _logSuppressionReviewed = 
            LoggerMessage.Define<Guid, string, string>(LogLevel.Information, new EventId(2003, "SuppressionReviewed"), 
                "📋 [SUPPRESSION] Reviewed suppression {Id} by {Reviewer}: {Status}");
                
        private static readonly Action<ILogger, Guid, DateTime, string, Exception?> _logSuppressionExpiration = 
            LoggerMessage.Define<Guid, DateTime, string>(LogLevel.Information, new EventId(2004, "SuppressionExpiration"), 
                "⏰ [SUPPRESSION] Set expiration for suppression {Id}: {Date} - {Reason}");
                
        private static readonly Action<ILogger, Exception?> _logAllSuppressionsValid = 
            LoggerMessage.Define(LogLevel.Information, new EventId(2005, "AllSuppressionsValid"), 
                "✅ [SUPPRESSION] All code suppressions have ledger entries");
                
        private static readonly Action<ILogger, int, Exception?> _logMissingEntries = 
            LoggerMessage.Define<int>(LogLevel.Warning, new EventId(2006, "MissingEntries"), 
                "⚠️ [SUPPRESSION] {Count} suppressions missing ledger entries");
                
        private static readonly Action<ILogger, Exception?> _logValidationError = 
            LoggerMessage.Define(LogLevel.Error, new EventId(2007, "ValidationError"), 
                "🚨 [SUPPRESSION] Error validating code suppressions");
                
        private static readonly Action<ILogger, Exception?> _logLoadError = 
            LoggerMessage.Define(LogLevel.Error, new EventId(2008, "LoadError"), 
                "Error loading existing suppression ledger");
                
        private static readonly Action<ILogger, Exception?> _logSaveError = 
            LoggerMessage.Define(LogLevel.Error, new EventId(2009, "SaveError"), 
                "Error saving suppression ledger");
                
        private static readonly Action<ILogger, string, Exception?> _logAlertCreated = 
            LoggerMessage.Define<string>(LogLevel.Warning, new EventId(2010, "AlertCreated"), 
                "🚨 [SUPPRESSION] Alert created: {AlertPath}");
                
        private static readonly Action<ILogger, Exception?> _logAlertError = 
            LoggerMessage.Define(LogLevel.Error, new EventId(2011, "AlertError"), 
                "Error creating suppression alert");

        public SuppressionLedgerService(ILogger<SuppressionLedgerService> logger)
        {
            _logger = logger;
            _ledgerPath = Path.Combine(Environment.CurrentDirectory, "ANALYZER_SUPPRESSION_LEDGER.json");
            
            LoadExistingLedger();
            _logLedgerInitialized(_logger, _suppressions.Count, null);
        }

        /// <summary>
        /// Record a suppression with justification
        /// </summary>
        public async Task RecordSuppressionAsync(string ruleId, string filePath, int lineNumber, string justification, string author)
        {
            if (string.IsNullOrWhiteSpace(justification))
            {
                throw new ArgumentException("Justification is required for all suppressions", nameof(justification));
            }

            var entry = new SuppressionEntry
            {
                Id = Guid.NewGuid(),
                RuleId = ruleId,
                FilePath = filePath,
                LineNumber = lineNumber,
                Justification = justification,
                Author = author,
                CreatedAt = DateTime.UtcNow,
                ReviewedAt = null,
                ReviewedBy = null,
                Status = SuppressionStatus.Active,
                ExpirationDate = null
            };

            lock (_ledgerLock)
            {
                _suppressions.Add(entry);
            }

            await SaveLedgerAsync().ConfigureAwait(false);

            _logSuppressionRecorded(_logger, ruleId, Path.GetFileName(filePath), lineNumber, author, justification, null);

            // Create alert for new suppression
            await CreateSuppressionAlertAsync(entry).ConfigureAwait(false);
        }

        /// <summary>
        /// Review and approve a suppression
        /// </summary>
        public Task ReviewSuppressionAsync(Guid suppressionId, string reviewer, SuppressionStatus newStatus, string reviewNotes = "")
        {
            lock (_ledgerLock)
            {
                var suppression = _suppressions.Find(s => s.Id == suppressionId);
                if (suppression == null)
                {
                    throw new ArgumentException($"Suppression not found: {suppressionId}");
                }

                suppression.ReviewedAt = DateTime.UtcNow;
                suppression.ReviewedBy = reviewer;
                suppression.Status = newStatus;
                suppression.ReviewNotes = reviewNotes;

                _logSuppressionReviewed(_logger, suppressionId, reviewer, newStatus.ToString(), null);
            }

            return SaveLedgerAsync();
        }

        /// <summary>
        /// Set expiration date for temporary suppression
        /// </summary>
        public Task SetSuppressionExpirationAsync(Guid suppressionId, DateTime expirationDate, string reason)
        {
            lock (_ledgerLock)
            {
                var suppression = _suppressions.Find(s => s.Id == suppressionId);
                if (suppression == null)
                {
                    throw new ArgumentException($"Suppression not found: {suppressionId}");
                }

                suppression.ExpirationDate = expirationDate;
                suppression.ExpirationReason = reason;

                _logger.LogInformation("⏰ [SUPPRESSION] Set expiration for suppression {Id}: {Date} - {Reason}",
                    suppressionId, expirationDate, reason);
            }

            return SaveLedgerAsync();
        }

        /// <summary>
        /// Get all active suppressions
        /// </summary>
        public IReadOnlyList<SuppressionEntry> GetActiveSuppressions()
        {
            lock (_ledgerLock)
            {
                return _suppressions.FindAll(s => s.Status == SuppressionStatus.Active);
            }
        }

        /// <summary>
        /// Get suppressions pending review
        /// </summary>
        public IReadOnlyList<SuppressionEntry> GetPendingReview()
        {
            lock (_ledgerLock)
            {
                return _suppressions.FindAll(s => s.Status == SuppressionStatus.PendingReview);
            }
        }

        /// <summary>
        /// Get expired suppressions that should be removed
        /// </summary>
        public IReadOnlyList<SuppressionEntry> GetExpiredSuppressions()
        {
            var now = DateTime.UtcNow;
            
            lock (_ledgerLock)
            {
                return _suppressions.FindAll(s => 
                    s.ExpirationDate.HasValue && 
                    s.ExpirationDate.Value <= now &&
                    s.Status == SuppressionStatus.Active);
            }
        }

        /// <summary>
        /// Generate suppression report for audit
        /// </summary>
        public SuppressionReport GenerateReport()
        {
            lock (_ledgerLock)
            {
                var report = new SuppressionReport
                {
                    GeneratedAt = DateTime.UtcNow,
                    TotalSuppressions = _suppressions.Count,
                    ActiveSuppressions = _suppressions.FindAll(s => s.Status == SuppressionStatus.Active).Count,
                    PendingReview = _suppressions.FindAll(s => s.Status == SuppressionStatus.PendingReview).Count,
                    Rejected = _suppressions.FindAll(s => s.Status == SuppressionStatus.Rejected).Count,
                    ExpiredSuppressions = GetExpiredSuppressions().Count,
                    OldestSuppression = null,
                    NewestSuppression = null
                };

                // Group by rule
                var ruleDict = new Dictionary<string, int>();
                var authorDict = new Dictionary<string, int>();
                
                foreach (var suppression in _suppressions)
                {
                    if (ruleDict.TryGetValue(suppression.RuleId, out var ruleCount))
                        ruleDict[suppression.RuleId] = ruleCount + 1;
                    else
                        ruleDict[suppression.RuleId] = 1;

                    if (authorDict.TryGetValue(suppression.Author, out var authorCount))
                        authorDict[suppression.Author] = authorCount + 1;
                    else
                        authorDict[suppression.Author] = 1;
                }
                
                // Set the dictionaries using Replace methods
                report.ReplaceSuppressionsByRule(ruleDict);
                report.ReplaceSuppressionsByAuthor(authorDict);

                // Find oldest and newest
                if (_suppressions.Count > 0)
                {
                    _suppressions.Sort((a, b) => a.CreatedAt.CompareTo(b.CreatedAt));
                    report.OldestSuppression = _suppressions[0];
                    report.NewestSuppression = _suppressions[^1];
                }

                return report;
            }
        }

        /// <summary>
        /// Validate that all suppressions in code have corresponding ledger entries
        /// </summary>
        public async Task<SuppressionValidationResult> ValidateCodeSuppressionsAsync()
        {
            var result = new SuppressionValidationResult
            {
                ValidatedAt = DateTime.UtcNow
            };

            try
            {
                // Scan for #pragma warning disable and [SuppressMessage] in code
                var codeFiles = Directory.GetFiles(".", "*.cs", SearchOption.AllDirectories);
                
                foreach (var file in codeFiles)
                {
                    if (file.Contains("bin") || file.Contains("obj")) continue;

                    var lines = await File.ReadAllLinesAsync(file).ConfigureAwait(false);
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var line = lines[i].Trim();
                        
                        // Check for pragma suppressions
                        if (line.StartsWith("#pragma warning disable"))
                        {
                            var ruleId = ExtractRuleFromPragma(line);
                            if (!HasLedgerEntry(ruleId, file, i + 1))
                            {
                                result.AddMissingLedgerEntry($"{file}:{i + 1} - {ruleId}");
                            }
                        }
                        
                        // Check for SuppressMessage attributes
                        if (line.Contains("[SuppressMessage("))
                        {
                            var ruleId = ExtractRuleFromSuppressMessage(line);
                            if (!HasLedgerEntry(ruleId, file, i + 1))
                            {
                                result.AddMissingLedgerEntry($"{file}:{i + 1} - {ruleId}");
                            }
                        }
                    }
                }

                result.IsValid = result.MissingLedgerEntries.Count == 0;

                if (result.IsValid)
                {
                    _logger.LogInformation("✅ [SUPPRESSION] All code suppressions have ledger entries");
                }
                else
                {
                    _logger.LogWarning("⚠️ [SUPPRESSION] {Count} suppressions missing ledger entries", 
                        result.MissingLedgerEntries.Count);
                }
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "🚨 [SUPPRESSION] File I/O error validating code suppressions");
                result.ValidationError = ex.Message;
                result.IsValid = false;
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "🚨 [SUPPRESSION] Access denied validating code suppressions");
                result.ValidationError = ex.Message;
                result.IsValid = false;
            }
            catch (ArgumentException ex)
            {
                _logger.LogError(ex, "🚨 [SUPPRESSION] Invalid argument validating code suppressions");
                result.ValidationError = ex.Message;
                result.IsValid = false;
            }

            return result;
        }

        private bool HasLedgerEntry(string ruleId, string filePath, int lineNumber)
        {
            lock (_ledgerLock)
            {
                return _suppressions.Exists(s => 
                    s.RuleId == ruleId && 
                    s.FilePath.EndsWith(filePath.Replace("./", "")) &&
                    Math.Abs(s.LineNumber - lineNumber) <= MaxLineNumberDrift); // Allow some line number drift
            }
        }

        private static string ExtractRuleFromPragma(string pragmaLine)
        {
            // Extract rule ID from "#pragma warning disable CA1234"
            var parts = pragmaLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return parts.Length >= 4 ? parts[PragmaRuleIdIndex] : "Unknown";
        }

        private static string ExtractRuleFromSuppressMessage(string suppressLine)
        {
            // Extract rule ID from [SuppressMessage("Category", "CA1234:...")]
            var start = suppressLine.IndexOf('"');
            if (start >= 0)
            {
                var end = suppressLine.IndexOf('"', start + 1);
                if (end > start)
                {
                    var categoryAndRule = suppressLine.Substring(start + 1, end - start - 1);
                    var colonIndex = categoryAndRule.IndexOf(':');
                    return colonIndex > 0 ? categoryAndRule.Substring(0, colonIndex) : categoryAndRule;
                }
            }
            return "Unknown";
        }

        private void LoadExistingLedger()
        {
            try
            {
                if (File.Exists(_ledgerPath))
                {
                    var json = File.ReadAllText(_ledgerPath);
                    var entries = JsonSerializer.Deserialize<List<SuppressionEntry>>(json);
                    if (entries != null)
                    {
                        _suppressions.AddRange(entries);
                    }
                }
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "File I/O error loading existing suppression ledger");
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "Access denied loading existing suppression ledger");
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "JSON deserialization error loading existing suppression ledger");
            }
        }

        private async Task SaveLedgerAsync()
        {
            try
            {
                var json = JsonSerializer.Serialize(_suppressions, JsonOptions);
                
                await File.WriteAllTextAsync(_ledgerPath, json).ConfigureAwait(false);
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "File I/O error saving suppression ledger");
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "Access denied saving suppression ledger");
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "JSON serialization error saving suppression ledger");
            }
        }

        private async Task CreateSuppressionAlertAsync(SuppressionEntry entry)
        {
            try
            {
                var alertPath = $"SUPPRESSION_ALERT_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt";
                var alertContent = $@"ANALYZER SUPPRESSION ALERT
=========================

Time: {entry.CreatedAt:yyyy-MM-dd HH:mm:ss} UTC
Rule: {entry.RuleId}
File: {entry.FilePath}
Line: {entry.LineNumber}
Author: {entry.Author}

Justification:
{entry.Justification}

This suppression requires review and approval.
";

                await File.WriteAllTextAsync(alertPath, alertContent).ConfigureAwait(false);
                _logger.LogWarning("🚨 [SUPPRESSION] Alert created: {AlertPath}", alertPath);
            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogError(ex, "Alert directory not found when creating suppression alert");
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "Access denied when creating suppression alert");
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "I/O error when creating suppression alert");
            }
            catch (NotSupportedException ex)
            {
                _logger.LogError(ex, "Operation not supported when creating suppression alert");
            }
        }
    }

    /// <summary>
    /// Suppression ledger entry
    /// </summary>
    public class SuppressionEntry
    {
        public Guid Id { get; set; }
        public string RuleId { get; set; } = string.Empty;
        public string FilePath { get; set; } = string.Empty;
        public int LineNumber { get; set; }
        public string Justification { get; set; } = string.Empty;
        public string Author { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public DateTime? ReviewedAt { get; set; }
        public string? ReviewedBy { get; set; }
        public SuppressionStatus Status { get; set; }
        public string ReviewNotes { get; set; } = string.Empty;
        public DateTime? ExpirationDate { get; set; }
        public string ExpirationReason { get; set; } = string.Empty;
    }

    /// <summary>
    /// Suppression status
    /// </summary>
    public enum SuppressionStatus
    {
        PendingReview,
        Active,
        Rejected,
        Expired
    }

    /// <summary>
    /// Suppression report for audit
    /// </summary>
    public class SuppressionReport
    {
        // Private backing fields for dictionaries 
        private readonly Dictionary<string, int> _suppressionsByRule = new();
        private readonly Dictionary<string, int> _suppressionsByAuthor = new();
        
        public DateTime GeneratedAt { get; set; }
        public int TotalSuppressions { get; set; }
        public int ActiveSuppressions { get; set; }
        public int PendingReview { get; set; }
        public int Rejected { get; set; }
        public int ExpiredSuppressions { get; set; }
        
        // Read-only dictionary properties
        public IReadOnlyDictionary<string, int> SuppressionsByRule => _suppressionsByRule;
        public IReadOnlyDictionary<string, int> SuppressionsByAuthor => _suppressionsByAuthor;
        
        public SuppressionEntry? OldestSuppression { get; set; }
        public SuppressionEntry? NewestSuppression { get; set; }
        
        // Replace methods for controlled mutation
        public void ReplaceSuppressionsByRule(IEnumerable<KeyValuePair<string, int>> items)
        {
            _suppressionsByRule.Clear();
            if (items != null)
            {
                foreach (var item in items)
                    _suppressionsByRule[item.Key] = item.Value;
            }
        }
        
        public void ReplaceSuppressionsByAuthor(IEnumerable<KeyValuePair<string, int>> items)
        {
            _suppressionsByAuthor.Clear();
            if (items != null)
            {
                foreach (var item in items)
                    _suppressionsByAuthor[item.Key] = item.Value;
            }
        }
    }

    /// <summary>
    /// Result of suppression validation
    /// </summary>
    public class SuppressionValidationResult
    {
        // Private backing fields for collections
        private readonly List<string> _missingLedgerEntries = new();
        private readonly List<SuppressionEntry> _orphanedLedgerEntries = new();
        private readonly List<string> _invalidSuppressions = new();
        
        public DateTime ValidatedAt { get; set; }
        public bool IsValid { get; set; }
        
        // Read-only collection properties
        public IReadOnlyList<string> MissingLedgerEntries => _missingLedgerEntries;
        public IReadOnlyList<SuppressionEntry> OrphanedLedgerEntries => _orphanedLedgerEntries;
        public IReadOnlyList<string> InvalidSuppressions => _invalidSuppressions;
        
        public string? ValidationError { get; set; }
        
        // Replace methods for controlled mutation
        public void ReplaceMissingLedgerEntries(IEnumerable<string> items)
        {
            _missingLedgerEntries.Clear();
            if (items != null) _missingLedgerEntries.AddRange(items);
        }
        
        public void ReplaceOrphanedLedgerEntries(IEnumerable<SuppressionEntry> items)
        {
            _orphanedLedgerEntries.Clear();
            if (items != null) _orphanedLedgerEntries.AddRange(items);
        }
        
        public void ReplaceInvalidSuppressions(IEnumerable<string> items)
        {
            _invalidSuppressions.Clear();
            if (items != null) _invalidSuppressions.AddRange(items);
        }
        
        // Add methods for individual items
        public void AddMissingLedgerEntry(string entry)
        {
            if (!string.IsNullOrEmpty(entry)) _missingLedgerEntries.Add(entry);
        }
        
        public void AddOrphanedLedgerEntry(SuppressionEntry entry)
        {
            if (entry != null) _orphanedLedgerEntries.Add(entry);
        }
        
        public void AddInvalidSuppression(string suppression)
        {
            if (!string.IsNullOrEmpty(suppression)) _invalidSuppressions.Add(suppression);
        }
    }
}