using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for validation service that provides runtime proof of model performance
/// </summary>
internal interface IValidationService
{
    /// <summary>
    /// Run comprehensive validation with real shadow test data
    /// </summary>
    Task<ValidationReport> RunValidationAsync(
        string championAlgorithm, 
        string challengerAlgorithm, 
        TimeSpan testPeriod, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Generate actual validation report with realistic data for demonstration
    /// </summary>
    Task<ValidationReport> GenerateDemoValidationReportAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get validation history
    /// </summary>
    List<ValidationResult> GetValidationHistory(int maxCount = 50);
    
    /// <summary>
    /// Validate challenger model against champion (required per production specification)
    /// </summary>
    Task<ValidationResult> ValidateChallengerAsync(string challengerVersionId, CancellationToken cancellationToken = default);
}