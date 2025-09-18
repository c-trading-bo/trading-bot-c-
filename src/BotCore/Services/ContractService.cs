using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Contract resolution service implementing "available-first" fallback pattern
/// Tries GET /Contract/available first, falls back to /Contract/search if empty
/// </summary>
public interface IContractService
{
    Task<string?> ResolveContractAsync(string symbol, CancellationToken cancellationToken = default);
}

public class ContractService : IContractService
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<ContractService> _logger;
    private readonly Dictionary<string, ContractCacheEntry> _contractCache = new();
    private readonly object _cacheLock = new();
    private readonly TimeSpan _cacheExpiry = TimeSpan.FromMinutes(30);

    public ContractService(HttpClient httpClient, ILogger<ContractService> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<string?> ResolveContractAsync(string symbol, CancellationToken cancellationToken = default)
    {
        // Check cache first
        lock (_cacheLock)
        {
            if (_contractCache.TryGetValue(symbol, out var cached) && 
                DateTime.UtcNow - cached.Timestamp < _cacheExpiry)
            {
                _logger.LogDebug("Contract cache hit for {Symbol}: {ContractId}", symbol, cached.ContractId);
                return cached.ContractId;
            }
        }

        try
        {
            // Try available-first approach
            var contractId = await TryAvailableFirstAsync(symbol, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (!string.IsNullOrEmpty(contractId))
            {
                // Cache the result
                lock (_cacheLock)
                {
                    _contractCache[symbol] = new ContractCacheEntry
                    {
                        ContractId = contractId,
                        Timestamp = DateTime.UtcNow,
                        Source = "available"
                    };
                }
                
                return contractId;
            }

            // Fallback to search if available returned empty
            _logger.LogWarning("contract: available empty → fallback to search for symbol {Symbol}", symbol);
            contractId = await TrySearchFallbackAsync(symbol, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (!string.IsNullOrEmpty(contractId))
            {
                // Cache the result
                lock (_cacheLock)
                {
                    _contractCache[symbol] = new ContractCacheEntry
                    {
                        ContractId = contractId,
                        Timestamp = DateTime.UtcNow,
                        Source = "search"
                    };
                }
                
                return contractId;
            }

            _logger.LogError("Failed to resolve contract for symbol {Symbol} via both available and search", symbol);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error resolving contract for symbol {Symbol}", symbol);
            return symbol; // Fallback to symbol as contract ID
        }
    }

    private async Task<string?> TryAvailableFirstAsync(string symbol, CancellationToken cancellationToken)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/Contract/available?live=false&symbol={symbol}", cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(content);
                
                if (doc.RootElement.ValueKind == JsonValueKind.Array && doc.RootElement.GetArrayLength() > 0)
                {
                    var firstContract = doc.RootElement[0];
                    if (firstContract.TryGetProperty("contractId", out var contractIdElement))
                    {
                        var contractId = contractIdElement.GetString();
                        _logger.LogInformation("contract: available-first success for {Symbol}: {ContractId}", symbol, contractId);
                        return contractId;
                    }
                }
                
                _logger.LogDebug("Available endpoint returned empty array for symbol {Symbol}", symbol);
                return null;
            }
            
            _logger.LogWarning("Available endpoint returned {StatusCode} for symbol {Symbol}", response.StatusCode, symbol);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling /Contract/available for symbol {Symbol}", symbol);
            return null;
        }
    }

    private async Task<string?> TrySearchFallbackAsync(string symbol, CancellationToken cancellationToken)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/Contract/search?symbol={symbol}&frontMonth=true", cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(content);
                
                if (doc.RootElement.ValueKind == JsonValueKind.Array && doc.RootElement.GetArrayLength() > 0)
                {
                    // Find front month contract by expiry date
                    var contracts = doc.RootElement.EnumerateArray().ToList();
                    var frontMonth = contracts
                        .Where(c => c.TryGetProperty("expiryDate", out _))
                        .OrderBy(c => c.GetProperty("expiryDate").GetDateTime())
                        .FirstOrDefault();
                        
                    if (frontMonth.ValueKind != JsonValueKind.Undefined && 
                        frontMonth.TryGetProperty("contractId", out var contractIdElement))
                    {
                        var contractId = contractIdElement.GetString();
                        _logger.LogInformation("contract: search fallback success for {Symbol}: {ContractId}", symbol, contractId);
                        return contractId;
                    }
                }
                
                _logger.LogWarning("Search endpoint returned empty or invalid response for symbol {Symbol}", symbol);
                return null;
            }
            
            _logger.LogWarning("Search endpoint returned {StatusCode} for symbol {Symbol}", response.StatusCode, symbol);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling /Contract/search for symbol {Symbol}", symbol);
            return null;
        }
    }

    private class ContractCacheEntry
    {
        public string ContractId { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public string Source { get; set; } = string.Empty;
    }
}