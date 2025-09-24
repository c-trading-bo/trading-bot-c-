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
    /// </summary>
    public class SuppressionLedgerService
    {
        private readonly ILogger<SuppressionLedgerService> _logger;
        private readonly string _ledgerPath;
        private readonly List<SuppressionEntry> _suppressions = new();
        private readonly object _ledgerLock = new();

        public SuppressionLedgerService(ILogger<SuppressionLedgerService> logger)
        {
            _logger = logger;
            _ledgerPath = Path.Combine(Environment.CurrentDirectory, "ANALYZER_SUPPRESSION_LEDGER.json");
            
            LoadExistingLedger();
            _logger.LogInformation("📋 [SUPPRESSION] Suppression ledger initialized with {Count} entries", _suppressions.Count);
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

            _logger.LogWarning("⚠️ [SUPPRESSION] Recorded suppression {RuleId} in {File}:{Line} by {Author}: {Justification}",
                ruleId, Path.GetFileName(filePath), lineNumber, author, justification);

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

                _logger.LogInformation("📋 [SUPPRESSION] Reviewed suppression {Id} by {Reviewer}: {Status}",
                    suppressionId, reviewer, newStatus);
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
        public List<SuppressionEntry> GetActiveSuppressions()
        {
            lock (_ledgerLock)
            {
                return _suppressions.FindAll(s => s.Status == SuppressionStatus.Active);
            }
        }

        /// <summary>
        /// Get suppressions pending review
        /// </summary>
        public List<SuppressionEntry> GetPendingReview()
        {
            lock (_ledgerLock)
            {
                return _suppressions.FindAll(s => s.Status == SuppressionStatus.PendingReview);
            }
        }

        /// <summary>
        /// Get expired suppressions that should be removed
        /// </summary>
        public List<SuppressionEntry> GetExpiredSuppressions()
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
                    SuppressionsByRule = new Dictionary<string, int>(),
                    SuppressionsByAuthor = new Dictionary<string, int>(),
                    OldestSuppression = null,
                    NewestSuppression = null
                };

                // Group by rule
                foreach (var suppression in _suppressions)
                {
                    if (report.SuppressionsByRule.ContainsKey(suppression.RuleId))
                        report.SuppressionsByRule[suppression.RuleId]++;
                    else
                        report.SuppressionsByRule[suppression.RuleId] = 1;

                    if (report.SuppressionsByAuthor.ContainsKey(suppression.Author))
                        report.SuppressionsByAuthor[suppression.Author]++;
                    else
                        report.SuppressionsByAuthor[suppression.Author] = 1;
                }

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
                ValidatedAt = DateTime.UtcNow,
                MissingLedgerEntries = new List<string>(),
                OrphanedLedgerEntries = new List<SuppressionEntry>(),
                InvalidSuppressions = new List<string>()
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
                                result.MissingLedgerEntries.Add($"{file}:{i + 1} - {ruleId}");
                            }
                        }
                        
                        // Check for SuppressMessage attributes
                        if (line.Contains("[SuppressMessage("))
                        {
                            var ruleId = ExtractRuleFromSuppressMessage(line);
                            if (!HasLedgerEntry(ruleId, file, i + 1))
                            {
                                result.MissingLedgerEntries.Add($"{file}:{i + 1} - {ruleId}");
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
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [SUPPRESSION] Error validating code suppressions");
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
                    Math.Abs(s.LineNumber - lineNumber) <= 2); // Allow some line number drift
            }
        }

        private string ExtractRuleFromPragma(string pragmaLine)
        {
            // Extract rule ID from "#pragma warning disable CA1234"
            var parts = pragmaLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return parts.Length >= 4 ? parts[3] : "Unknown";
        }

        private string ExtractRuleFromSuppressMessage(string suppressLine)
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
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading existing suppression ledger");
            }
        }

        private async Task SaveLedgerAsync()
        {
            try
            {
                var json = JsonSerializer.Serialize(_suppressions, new JsonSerializerOptions 
                { 
                    WriteIndented = true,
                    Converters = { new JsonStringEnumConverter() }
                });
                
                await File.WriteAllTextAsync(_ledgerPath, json).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving suppression ledger");
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
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating suppression alert");
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
        public DateTime GeneratedAt { get; set; }
        public int TotalSuppressions { get; set; }
        public int ActiveSuppressions { get; set; }
        public int PendingReview { get; set; }
        public int Rejected { get; set; }
        public int ExpiredSuppressions { get; set; }
        public Dictionary<string, int> SuppressionsByRule { get; set; } = new();
        public Dictionary<string, int> SuppressionsByAuthor { get; set; } = new();
        public SuppressionEntry? OldestSuppression { get; set; }
        public SuppressionEntry? NewestSuppression { get; set; }
    }

    /// <summary>
    /// Result of suppression validation
    /// </summary>
    public class SuppressionValidationResult
    {
        public DateTime ValidatedAt { get; set; }
        public bool IsValid { get; set; }
        public List<string> MissingLedgerEntries { get; set; } = new();
        public List<SuppressionEntry> OrphanedLedgerEntries { get; set; } = new();
        public List<string> InvalidSuppressions { get; set; } = new();
        public string? ValidationError { get; set; }
    }
}