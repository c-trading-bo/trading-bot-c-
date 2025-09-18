using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Collections.Concurrent;
using TradingBot.Abstractions;

namespace Trading.Safety.Journaling;

/// <summary>
/// Production-grade trade journaling system with tamper-evident logging
/// Persists all decision→order→fill→outcome events in immutable, auditable format
/// </summary>
public interface ITradeJournal
{
    Task LogDecisionAsync(TradingDecisionEvent decisionEvent);
    Task LogOrderAsync(OrderEvent orderEvent);
    Task LogFillAsync(FillEvent fillEvent);
    Task LogOutcomeAsync(OutcomeEvent outcomeEvent);
    Task<List<TradeJournalEntry>> GetTradesAsync(DateTime from, DateTime to);
    Task<TradeJournalEntry?> GetTradeByIdAsync(string tradeId);
    Task ValidateIntegrityAsync();
    Task ArchiveAsync(DateTime before);
}

public class TradeJournal : ITradeJournal
{
    private readonly ILogger<TradeJournal> _logger;
    private readonly TradeJournalConfig _config;
    private readonly string _journalDirectory;
    private readonly ConcurrentDictionary<string, TradeJournalEntry> _activeEntries = new();
    private readonly object _writeLock = new object();

    public TradeJournal(
        ILogger<TradeJournal> logger,
        IOptions<TradeJournalConfig> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        _journalDirectory = _config.JournalDirectory ?? Path.Combine(Environment.CurrentDirectory, "journal");
        
        Directory.CreateDirectory(_journalDirectory);
        Directory.CreateDirectory(Path.Combine(_journalDirectory, "archive"));
    }

    public async Task LogDecisionAsync(TradingDecisionEvent decisionEvent)
    {
        var entry = new TradeJournalEntry
        {
            TradeId = decisionEvent.TradeId,
            CorrelationId = decisionEvent.CorrelationId,
            CreatedAt = DateTime.UtcNow,
            Decision = decisionEvent,
            Status = TradeJournalStatus.DecisionMade
        };

        _activeEntries.AddOrUpdate(entry.TradeId, entry, (k, existing) =>
        {
            existing.Decision = decisionEvent;
            existing.Status = TradeJournalStatus.DecisionMade;
            return existing;
        });

        await WriteJournalEntryAsync(entry, "DECISION").ConfigureAwait(false);
        _logger.LogInformation("[TRADE_JOURNAL] Decision logged: {TradeId} {Symbol} {Side} {Quantity} [CorrelationId: {CorrelationId}]",
            entry.TradeId, decisionEvent.Symbol, decisionEvent.Side, decisionEvent.Quantity, entry.CorrelationId);
    }

    public async Task LogOrderAsync(OrderEvent orderEvent)
    {
        if (_activeEntries.TryGetValue(orderEvent.TradeId, out var entry))
        {
            entry.Order = orderEvent;
            entry.Status = TradeJournalStatus.OrderPlaced;
            entry.UpdatedAt = DateTime.UtcNow;

            await WriteJournalEntryAsync(entry, "ORDER").ConfigureAwait(false);
            _logger.LogInformation("[TRADE_JOURNAL] Order logged: {TradeId} {OrderId} {Status} [CorrelationId: {CorrelationId}]",
                entry.TradeId, orderEvent.OrderId, orderEvent.Status, entry.CorrelationId);
        }
        else
        {
            _logger.LogWarning("[TRADE_JOURNAL] Order event for unknown trade: {TradeId} {OrderId}", 
                orderEvent.TradeId, orderEvent.OrderId);
        }
    }

    public async Task LogFillAsync(FillEvent fillEvent)
    {
        if (_activeEntries.TryGetValue(fillEvent.TradeId, out var entry))
        {
            entry.Fills ??= new List<FillEvent>();
            entry.Fills.Add(fillEvent);
            entry.Status = TradeJournalStatus.PartiallyFilled;
            entry.UpdatedAt = DateTime.UtcNow;

            // Check if fully filled
            var totalFilled = entry.Fills.Sum(f => f.FilledQuantity);
            if (entry.Decision != null && totalFilled >= entry.Decision.Quantity)
            {
                entry.Status = TradeJournalStatus.FullyFilled;
            }

            await WriteJournalEntryAsync(entry, "FILL").ConfigureAwait(false);
            _logger.LogInformation("[TRADE_JOURNAL] Fill logged: {TradeId} {FillId} {FilledQuantity}@{FillPrice} [CorrelationId: {CorrelationId}]",
                entry.TradeId, fillEvent.FillId, fillEvent.FilledQuantity, fillEvent.FillPrice, entry.CorrelationId);
        }
        else
        {
            _logger.LogWarning("[TRADE_JOURNAL] Fill event for unknown trade: {TradeId} {FillId}", 
                fillEvent.TradeId, fillEvent.FillId);
        }
    }

    public async Task LogOutcomeAsync(OutcomeEvent outcomeEvent)
    {
        if (_activeEntries.TryGetValue(outcomeEvent.TradeId, out var entry))
        {
            entry.Outcome = outcomeEvent;
            entry.Status = TradeJournalStatus.Completed;
            entry.CompletedAt = DateTime.UtcNow;
            entry.UpdatedAt = DateTime.UtcNow;

            await WriteJournalEntryAsync(entry, "OUTCOME").ConfigureAwait(false);
            _logger.LogInformation("[TRADE_JOURNAL] Outcome logged: {TradeId} {PnL} {Outcome} [CorrelationId: {CorrelationId}]",
                entry.TradeId, outcomeEvent.RealizedPnL, outcomeEvent.OutcomeType, entry.CorrelationId);

            // Move to completed entries after configurable delay
            _ = Task.Delay(TimeSpan.FromMinutes(5)).ContinueWith(t =>
            {
                _activeEntries.TryRemove(entry.TradeId, out _);
            });
        }
        else
        {
            _logger.LogWarning("[TRADE_JOURNAL] Outcome event for unknown trade: {TradeId}", 
                outcomeEvent.TradeId);
        }
    }

    public async Task<List<TradeJournalEntry>> GetTradesAsync(DateTime from, DateTime to)
    {
        var entries = new List<TradeJournalEntry>();
        
        // Search journal files within date range
        var journalFiles = Directory.GetFiles(_journalDirectory, "trade_journal_*.json")
            .Where(f => IsFileInDateRange(f, from, to))
            .OrderBy(f => f);

        foreach (var file in journalFiles)
        {
            try
            {
                var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                foreach (var line in lines)
                {
                    if (TryDeserializeJournalEntry(line, out var entry) &&
                        entry != null &&
                        entry.CreatedAt >= from && entry.CreatedAt <= to)
                    {
                        entries.Add(entry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TRADE_JOURNAL] Error reading journal file: {File}", file);
            }
        }

        return entries.OrderBy(e => e.CreatedAt).ToList();
    }

    public async Task<TradeJournalEntry?> GetTradeByIdAsync(string tradeId)
    {
        // Check active entries first
        if (_activeEntries.TryGetValue(tradeId, out var activeEntry))
        {
            return activeEntry;
        }

        // Search in journal files
        var journalFiles = Directory.GetFiles(_journalDirectory, "trade_journal_*.json")
            .OrderByDescending(f => f); // Search newest first

        foreach (var file in journalFiles)
        {
            try
            {
                var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                foreach (var line in lines.Reverse()) // Search newest entries first
                {
                    if (TryDeserializeJournalEntry(line, out var entry) &&
                        entry != null &&
                        entry.TradeId == tradeId)
                    {
                        return entry;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TRADE_JOURNAL] Error searching journal file: {File}", file);
            }
        }

        return null;
    }

    public async Task ValidateIntegrityAsync()
    {
        var journalFiles = Directory.GetFiles(_journalDirectory, "trade_journal_*.json");
        var totalEntries;
        var corruptedEntries;

        foreach (var file in journalFiles)
        {
            try
            {
                var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                foreach (var line in lines)
                {
                    totalEntries++;
                    
                    if (!ValidateJournalEntryIntegrity(line))
                    {
                        corruptedEntries++;
                        _logger.LogError("[TRADE_JOURNAL] Corrupted entry detected in {File}: {Line}", 
                            file, line[..Math.Min(100, line.Length)]);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TRADE_JOURNAL] Error validating file: {File}", file);
            }
        }

        if (corruptedEntries > 0)
        {
            _logger.LogCritical("[TRADE_JOURNAL] Integrity check failed: {CorruptedEntries}/{TotalEntries} entries corrupted", 
                corruptedEntries, totalEntries);
            throw new InvalidOperationException($"Trade journal integrity compromised: {corruptedEntries} corrupted entries");
        }

        _logger.LogInformation("[TRADE_JOURNAL] Integrity check passed: {TotalEntries} entries validated", totalEntries);
    }

    public async Task ArchiveAsync(DateTime before)
    {
        var archiveDir = Path.Combine(_journalDirectory, "archive");
        var journalFiles = Directory.GetFiles(_journalDirectory, "trade_journal_*.json");

        foreach (var file in journalFiles)
        {
            var fileDate = ExtractDateFromJournalFile(file);
            if (fileDate < before)
            {
                var archiveFile = Path.Combine(archiveDir, Path.GetFileName(file));
                File.Move(file, archiveFile);
                _logger.LogInformation("[TRADE_JOURNAL] Archived journal file: {File}", file);
            }
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    private async Task WriteJournalEntryAsync(TradeJournalEntry entry, string eventType)
    {
        var journalRecord = new TradeJournalRecord
        {
            Timestamp = DateTime.UtcNow,
            EventType = eventType,
            Entry = entry
        };

        var json = JsonSerializer.Serialize(journalRecord, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });

        // Calculate hash for tamper detection
        var hash = CalculateHash(json);
        var journalLine = $"{json}|{hash}";

        var fileName = $"trade_journal_{DateTime.UtcNow:yyyyMMdd}.json";
        var filePath = Path.Combine(_journalDirectory, fileName);

        lock (_writeLock)
        {
            File.AppendAllText(filePath, journalLine + Environment.NewLine);
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    private string CalculateHash(string input)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input + _config.HashSalt));
        return Convert.ToBase64String(hashBytes);
    }

    private bool ValidateJournalEntryIntegrity(string journalLine)
    {
        try
        {
            var parts = journalLine.Split('|');
            if (parts.Length != 2) return false;

            var json = parts[0];
            var providedHash = parts[1];
            var calculatedHash = CalculateHash(json);

            return providedHash == calculatedHash;
        }
        catch
        {
            return false;
        }
    }

    private bool TryDeserializeJournalEntry(string journalLine, out TradeJournalEntry? entry)
    {
        entry = null;
        try
        {
            if (!ValidateJournalEntryIntegrity(journalLine)) return false;

            var json = journalLine.Split('|')[0];
            var record = JsonSerializer.Deserialize<TradeJournalRecord>(json, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            entry = record?.Entry;
            return entry != null;
        }
        catch
        {
            return false;
        }
    }

    private bool IsFileInDateRange(string filePath, DateTime from, DateTime to)
    {
        var fileDate = ExtractDateFromJournalFile(filePath);
        return fileDate >= from.Date && fileDate <= to.Date;
    }

    private DateTime ExtractDateFromJournalFile(string filePath)
    {
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        var datePart = fileName.Replace("trade_journal_", "");
        return DateTime.TryParseExact(datePart, "yyyyMMdd", null, System.Globalization.DateTimeStyles.None, out var date) 
            ? date : DateTime.MinValue;
    }
}

// Data models for trade journaling
public class TradeJournalEntry
{
    public string TradeId { get; set; } = string.Empty;
    public string CorrelationId { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public DateTime? UpdatedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public TradeJournalStatus Status { get; set; }
    public TradingDecisionEvent? Decision { get; set; }
    public OrderEvent? Order { get; set; }
    public List<FillEvent>? Fills { get; set; }
    public OutcomeEvent? Outcome { get; set; }
}

public class TradeJournalRecord
{
    public DateTime Timestamp { get; set; }
    public string EventType { get; set; } = string.Empty;
    public TradeJournalEntry Entry { get; set; } = null!;
}

public enum TradeJournalStatus
{
    DecisionMade,
    OrderPlaced,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Rejected,
    Completed,
    Failed
}

// Event models
public class TradingDecisionEvent
{
    public string TradeId { get; set; } = string.Empty;
    public string CorrelationId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty; // BUY/SELL
    public decimal Quantity { get; set; }
    public decimal? LimitPrice { get; set; }
    public decimal? StopPrice { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public decimal Confidence { get; set; }
    public string Regime { get; set; } = string.Empty;
    public Dictionary<string, object> Context { get; } = new();
}

public class OrderEvent
{
    public string TradeId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Status { get; set; } = string.Empty; // NEW/OPEN/FILLED/CANCELLED/REJECTED
    public string? RejectReason { get; set; }
    public Dictionary<string, object> Metadata { get; } = new();
}

public class FillEvent
{
    public string TradeId { get; set; } = string.Empty;
    public string FillId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal FillPrice { get; set; }
    public decimal Commission { get; set; }
    public Dictionary<string, object> Metadata { get; } = new();
}

public class OutcomeEvent
{
    public string TradeId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public decimal RealizedPnL { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public string OutcomeType { get; set; } = string.Empty; // WIN/LOSS/BREAKEVEN
    public decimal RMultiple { get; set; }
    public TimeSpan HoldingPeriod { get; set; }
    public Dictionary<string, object> Analytics { get; } = new();
}

public class TradeJournalConfig
{
    public string JournalDirectory { get; set; } = "./journal";
    public string HashSalt { get; set; } = "TradingBot_Journal_Salt_2024";
    public TimeSpan ArchiveAfter { get; set; } = TimeSpan.FromDays(90);
    public bool EnableIntegrityChecks { get; set; } = true;
}