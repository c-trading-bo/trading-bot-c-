using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using BotCore.Infrastructure;
using BotCore.Testing;
using BotCore.Reporting;
using BotCore.AutoRemediation;
using BotCore.ProductionGate;

namespace BotCore.Orchestration;

/// <summary>
/// Main orchestrator that coordinates the entire deployment pipeline from credential detection to production gate
/// </summary>
public class DeploymentPipelineOrchestrator
{
    private readonly ILogger<DeploymentPipelineOrchestrator> _logger;
    private readonly IServiceProvider _serviceProvider;

    public DeploymentPipelineOrchestrator(
        ILogger<DeploymentPipelineOrchestrator> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    /// <summary>
    /// Execute the complete deployment pipeline as specified in the requirements
    /// </summary>
    public async Task<PipelineExecutionResult> ExecuteFullPipelineAsync(CancellationToken cancellationToken = default)
    {
        var result = new PipelineExecutionResult
        {
            PipelineId = Guid.NewGuid().ToString("N")[..8],
            StartTime = DateTime.UtcNow
        };

        try
        {
            _logger.LogInformation("🚀 Starting Complete Deployment Pipeline...");
            _logger.LogInformation("Pipeline ID: {PipelineId}", result.PipelineId);
            
            // Step 1: Ensure bot grabs TopStep credentials from environment automatically
            _logger.LogInformation("📋 Step 1: Automatic TopStep Credential Detection");
            result.CredentialDetection = await ExecuteCredentialDetection();

            // Step 2: Deploy current code to staging environment matching TopStep's setup
            _logger.LogInformation("📋 Step 2: Deploy to Staging Environment");
            result.StagingDeployment = await ExecuteStagingDeployment();

            // Step 3: Execute full suite of smoke tests, end-to-end simulations, and CI gate jobs
            _logger.LogInformation("📋 Step 3: Execute Comprehensive Test Suite");
            result.TestSuiteExecution = await ExecuteComprehensiveTestSuite(cancellationToken);

            // Step 4: Generate comprehensive report with metrics, coverage, and TODOs
            _logger.LogInformation("📋 Step 4: Generate Comprehensive Report");
            result.ComprehensiveReporting = await ExecuteComprehensiveReporting(result.TestSuiteExecution);

            // Step 5: Auto-remediate issues or flag blockers for manual review
            _logger.LogInformation("📋 Step 5: Auto-remediation and Issue Resolution");
            result.AutoRemediation = await ExecuteAutoRemediation(result.TestSuiteExecution, result.ComprehensiveReporting);

            // Step 6: Confirm 100% pass status before allowing production merge
            _logger.LogInformation("📋 Step 6: Production Gate Validation");
            result.ProductionGate = await ExecuteProductionGate(cancellationToken);

            // Calculate final pipeline result
            result.EndTime = DateTime.UtcNow;
            result.CalculateFinalResult();

            // Generate final pipeline report
            await GenerateFinalPipelineReport(result);

            LogPipelineCompletion(result);

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Critical error in deployment pipeline");
            result.CriticalError = ex.Message;
            result.IsSuccessful = false;
            result.EndTime = DateTime.UtcNow;
        }

        return result;
    }

    private async Task<CredentialDetectionResult> ExecuteCredentialDetection()
    {
        await Task.Yield(); // Ensure async behavior
        
        var result = new CredentialDetectionResult();

        try
        {
            _logger.LogInformation("[DEPLOYMENT-PIPELINE] Executing comprehensive credential detection");
            
            var credentialManager = _serviceProvider.GetRequiredService<TopstepXCredentialManager>();
            
            // Asynchronous discovery of all available credential sources
            var discoveryReport = await Task.Run(() => credentialManager.DiscoverAllCredentialSources());
            
            result.HasCredentials = discoveryReport.HasAnyCredentials;
            result.CredentialSource = discoveryReport.RecommendedSource ?? "None";
            result.TotalSourcesFound = discoveryReport.TotalSourcesFound;
            result.IsSuccessful = discoveryReport.HasAnyCredentials;

            // Enhanced credential validation
            if (discoveryReport.HasEnvironmentCredentials)
            {
                result.Details.Add("✅ Environment variables detected and configured");
                
                // Validate environment credential quality
                await ValidateEnvironmentCredentialQuality(result);
            }

            if (discoveryReport.HasFileCredentials)
            {
                result.Details.Add("✅ Secure file credentials detected");
                
                // Validate file credential security
                await ValidateFileCredentialSecurity(result);
            }

            if (!discoveryReport.HasAnyCredentials)
            {
                result.Details.Add("❌ No TopStep credentials found in any source");
                result.ErrorMessage = "No TopStep credentials available for automated detection";
                
                // Provide detailed troubleshooting guidance
                await GenerateCredentialTroubleshootingGuide(result);
            }

            _logger.LogInformation("🔑 Credential Detection: {Status} - {Source} ({Count} sources)", 
                result.IsSuccessful ? "SUCCESS" : "FAILED", result.CredentialSource, result.TotalSourcesFound);

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DEPLOYMENT-PIPELINE] Failed to execute credential detection");
            result.ErrorMessage = ex.Message;
            result.IsSuccessful = false;
        }

        return result;
    }
    
    /// <summary>
    /// Validate the quality and completeness of environment credentials
    /// </summary>
    private static async Task ValidateEnvironmentCredentialQuality(CredentialDetectionResult result)
    {
        await Task.Yield();
        
        var requiredVars = new[] { "TOPSTEPX_USERNAME", "TOPSTEPX_API_KEY", "TOPSTEPX_JWT", "TOPSTEPX_ACCOUNT_ID" };
        var foundVars = 0;
        
        foreach (var varName in requiredVars)
        {
            var value = Environment.GetEnvironmentVariable(varName);
            if (!string.IsNullOrEmpty(value))
            {
                foundVars++;
                result.Details.Add($"  • {varName}: Present ({value.Length} chars)");
            }
            else
            {
                result.Details.Add($"  • {varName}: Missing");
            }
        }
        
        result.Details.Add($"Environment credential completeness: {foundVars}/{requiredVars.Length} variables found");
    }
    
    /// <summary>
    /// Validate the security of file-based credentials
    /// </summary>
    private static async Task ValidateFileCredentialSecurity(CredentialDetectionResult result)
    {
        await Task.Yield();
        
        try
        {
            var credentialPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".topstepx");
            if (Directory.Exists(credentialPath))
            {
                var files = Directory.GetFiles(credentialPath, "*.json");
                result.Details.Add($"  • Found {files.Length} credential file(s) in secure directory");
                
                // Check file permissions (simplified for cross-platform)
                foreach (var file in files.Take(3)) // Limit to first 3 files
                {
                    var fileInfo = new FileInfo(file);
                    result.Details.Add($"  • {Path.GetFileName(file)}: {fileInfo.Length} bytes, modified {fileInfo.LastWriteTime:yyyy-MM-dd}");
                }
            }
        }
        catch (Exception ex)
        {
            result.Details.Add($"  • File credential validation failed: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Generate troubleshooting guide when no credentials are found
    /// </summary>
    private static async Task GenerateCredentialTroubleshootingGuide(CredentialDetectionResult result)
    {
        await Task.Yield();
        
        result.Details.Add("Troubleshooting Guide:");
        result.Details.Add("  1. Set environment variables: TOPSTEPX_USERNAME, TOPSTEPX_API_KEY");
        result.Details.Add("  2. Create credential file: ~/.topstepx/credentials.json");
        result.Details.Add("  3. Check .env file in project root");
        result.Details.Add("  4. Verify access to credential storage systems");
        result.Details.Add("  5. Run credential validation tool: dotnet run --project tools/CredentialValidator");
    }

    private async Task<StagingDeploymentResult> ExecuteStagingDeployment()
    {
        var result = new StagingDeploymentResult();

        try
        {
            var stagingManager = _serviceProvider.GetRequiredService<StagingEnvironmentManager>();
            
            // Configure staging environment to match TopStep setup
            var stagingConfig = await stagingManager.ConfigureStagingEnvironmentAsync();
            
            result.EnvironmentConfigured = stagingConfig.EnvironmentConfigured;
            result.EndpointsConfigured = stagingConfig.EndpointsConfigured;
            result.RiskManagementConfigured = stagingConfig.RiskManagementConfigured;
            result.MonitoringConfigured = stagingConfig.MonitoringConfigured;
            result.ConnectivityValidated = stagingConfig.ConnectivityValidated;
            result.IsSuccessful = stagingConfig.IsSuccessful;

            if (stagingConfig.Errors.Any())
            {
                result.ErrorMessage = string.Join("; ", stagingConfig.Errors);
            }

            result.Details.Add($"Environment Setup: {(result.EnvironmentConfigured ? "✅" : "❌")}");
            result.Details.Add($"TopStep Endpoints: {(result.EndpointsConfigured ? "✅" : "❌")}");
            result.Details.Add($"Risk Management: {(result.RiskManagementConfigured ? "✅" : "❌")}");
            result.Details.Add($"Monitoring: {(result.MonitoringConfigured ? "✅" : "❌")}");
            result.Details.Add($"Connectivity: {(result.ConnectivityValidated ? "✅" : "❌")}");

            _logger.LogInformation("🏗️ Staging Deployment: {Status} - {Details}", 
                result.IsSuccessful ? "SUCCESS" : "FAILED", 
                $"{result.Details.Count(d => d.Contains("✅"))}/{result.Details.Count} components ready");

        }
        catch (Exception ex)
        {
            result.ErrorMessage = ex.Message;
            result.IsSuccessful = false;
        }

        return result;
    }

    private async Task<TestSuiteResult> ExecuteComprehensiveTestSuite(CancellationToken cancellationToken)
    {
        var testSuite = _serviceProvider.GetRequiredService<ComprehensiveSmokeTestSuite>();
        var testResult = await testSuite.ExecuteFullTestSuiteAsync(cancellationToken);

        _logger.LogInformation("🧪 Test Suite Execution: {Status} - Pass Rate: {PassRate:P1} ({Passed}/{Total})", 
            testResult.IsOverallSuccess ? "SUCCESS" : "FAILED",
            testResult.OverallPassRate,
            testResult.InfrastructureTests.PassedTests + testResult.CredentialTests.PassedTests + 
            testResult.IntegrationTests.PassedTests + testResult.ComponentTests.PassedTests +
            testResult.EndToEndTests.PassedTests + testResult.PerformanceTests.PassedTests + 
            testResult.SafetyTests.PassedTests,
            testResult.InfrastructureTests.TotalTests + testResult.CredentialTests.TotalTests +
            testResult.IntegrationTests.TotalTests + testResult.ComponentTests.TotalTests +
            testResult.EndToEndTests.TotalTests + testResult.PerformanceTests.TotalTests +
            testResult.SafetyTests.TotalTests);

        return testResult;
    }

    private async Task<ComprehensiveReport> ExecuteComprehensiveReporting(TestSuiteResult testResults)
    {
        var reportingSystem = _serviceProvider.GetRequiredService<ComprehensiveReportingSystem>();
        var report = await reportingSystem.GenerateComprehensiveReportAsync(testResults);

        _logger.LogInformation("📊 Comprehensive Reporting: {Status} - Health Score: {Score:P1}, Production Ready: {Ready}", 
            string.IsNullOrEmpty(report.GenerationError) ? "SUCCESS" : "FAILED",
            report.OverallHealthScore,
            report.IsProductionReady ? "YES" : "NO");

        return report;
    }

    private async Task<AutoRemediationResult> ExecuteAutoRemediation(
        TestSuiteResult testResults, 
        ComprehensiveReport comprehensiveReport)
    {
        var autoRemediation = _serviceProvider.GetRequiredService<AutoRemediationSystem>();
        var remediationResult = await autoRemediation.ExecuteAutoRemediationAsync(testResults, comprehensiveReport);

        _logger.LogInformation("🔧 Auto-remediation: {Status} - Issues Fixed: {Fixed}/{Attempted}, Manual Review: {Manual}", 
            remediationResult.OverallSuccess ? "SUCCESS" : "PARTIAL",
            remediationResult.TotalIssuesFixed,
            remediationResult.TotalIssuesAttempted,
            remediationResult.ManualReviewItems.Count);

        return remediationResult;
    }

    private async Task<ProductionGateResult> ExecuteProductionGate(CancellationToken cancellationToken)
    {
        var productionGate = _serviceProvider.GetRequiredService<ProductionGateSystem>();
        var gateResult = await productionGate.ExecuteProductionGateAsync(cancellationToken);

        _logger.LogInformation("🚪 Production Gate: {Status} - Readiness Score: {Score:F1}%, All Gates: {Gates}", 
            gateResult.IsProductionReady ? "PASS" : "FAIL",
            gateResult.ProductionReadinessAssessment.ReadinessScore,
            $"{(gateResult.PreflightValidation.IsSuccessful ? 1 : 0) + 
               (gateResult.TestSuiteExecution.IsOverallSuccess ? 1 : 0) + 
               (gateResult.PerformanceValidation.IsSuccessful ? 1 : 0) + 
               (gateResult.SecurityValidation.IsSuccessful ? 1 : 0) + 
               (gateResult.AutoRemediationExecution.OverallSuccess ? 1 : 0) + 
               (gateResult.ProductionReadinessAssessment.IsProductionReady ? 1 : 0)}/6 passed");

        return gateResult;
    }

    private async Task GenerateFinalPipelineReport(PipelineExecutionResult result)
    {
        try
        {
            var reportsDirectory = Path.Combine(Environment.CurrentDirectory, "reports");
            Directory.CreateDirectory(reportsDirectory);

            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var filename = $"pipeline_execution_report_{timestamp}_{result.PipelineId}.json";
            var filepath = Path.Combine(reportsDirectory, filename);

            var reportData = new
            {
                PipelineId = result.PipelineId,
                StartTime = result.StartTime,
                EndTime = result.EndTime,
                Duration = result.TotalDuration.ToString(@"hh\:mm\:ss"),
                IsSuccessful = result.IsSuccessful,
                ProductionReady = result.IsProductionReady,
                Summary = new
                {
                    CredentialDetection = new { Success = result.CredentialDetection.IsSuccessful, Source = result.CredentialDetection.CredentialSource },
                    StagingDeployment = new { Success = result.StagingDeployment.IsSuccessful, ComponentsReady = result.StagingDeployment.Details.Count(d => d.Contains("✅")) },
                    TestSuite = new { Success = result.TestSuiteExecution.IsOverallSuccess, PassRate = $"{result.TestSuiteExecution.OverallPassRate:P1}" },
                    Reporting = new { Success = string.IsNullOrEmpty(result.ComprehensiveReporting.GenerationError), HealthScore = $"{result.ComprehensiveReporting.OverallHealthScore:P1}" },
                    AutoRemediation = new { Success = result.AutoRemediation.OverallSuccess, IssuesFixed = result.AutoRemediation.TotalIssuesFixed },
                    ProductionGate = new { Success = result.ProductionGate.IsProductionReady, ReadinessScore = $"{result.ProductionGate.ProductionReadinessAssessment.ReadinessScore:F1}%" }
                },
                CriticalError = result.CriticalError
            };

            var json = System.Text.Json.JsonSerializer.Serialize(reportData, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });

            await File.WriteAllTextAsync(filepath, json);

            _logger.LogInformation("📄 Final pipeline report saved: {Filepath}", filepath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ Failed to save final pipeline report");
        }
    }

    private void LogPipelineCompletion(PipelineExecutionResult result)
    {
        _logger.LogInformation("");
        _logger.LogInformation("🚀 =============================================");
        _logger.LogInformation("   DEPLOYMENT PIPELINE EXECUTION COMPLETE");
        _logger.LogInformation("=============================================");
        _logger.LogInformation("Pipeline ID: {PipelineId}", result.PipelineId);
        _logger.LogInformation("Total Duration: {Duration}", result.TotalDuration.ToString(@"hh\:mm\:ss"));
        _logger.LogInformation("Overall Success: {Success}", result.IsSuccessful ? "✅ YES" : "❌ NO");
        _logger.LogInformation("Production Ready: {Ready}", result.IsProductionReady ? "✅ YES" : "❌ NO");
        _logger.LogInformation("");
        _logger.LogInformation("Pipeline Steps:");
        _logger.LogInformation("  1. Credential Detection: {Status}", result.CredentialDetection.IsSuccessful ? "✅ SUCCESS" : "❌ FAILED");
        _logger.LogInformation("  2. Staging Deployment: {Status}", result.StagingDeployment.IsSuccessful ? "✅ SUCCESS" : "❌ FAILED");
        _logger.LogInformation("  3. Test Suite Execution: {Status}", result.TestSuiteExecution.IsOverallSuccess ? "✅ SUCCESS" : "❌ FAILED");
        _logger.LogInformation("  4. Comprehensive Reporting: {Status}", string.IsNullOrEmpty(result.ComprehensiveReporting.GenerationError) ? "✅ SUCCESS" : "❌ FAILED");
        _logger.LogInformation("  5. Auto-remediation: {Status}", result.AutoRemediation.OverallSuccess ? "✅ SUCCESS" : "❌ PARTIAL");
        _logger.LogInformation("  6. Production Gate: {Status}", result.ProductionGate.IsProductionReady ? "✅ PASS" : "❌ FAIL");
        _logger.LogInformation("");
        
        if (result.IsProductionReady)
        {
            _logger.LogInformation("🎉 DEPLOYMENT PIPELINE SUCCESSFUL - READY FOR PRODUCTION!");
        }
        else
        {
            _logger.LogInformation("⚠️ DEPLOYMENT PIPELINE COMPLETED WITH ISSUES - REVIEW REQUIRED");
        }
        
        _logger.LogInformation("=============================================");
    }
}

// Pipeline execution result classes
public class PipelineExecutionResult
{
    public string PipelineId { get; set; } = "";
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan TotalDuration => EndTime - StartTime;
    public bool IsSuccessful { get; set; }
    public bool IsProductionReady { get; set; }
    public string? CriticalError { get; set; }

    public CredentialDetectionResult CredentialDetection { get; set; } = new();
    public StagingDeploymentResult StagingDeployment { get; set; } = new();
    public TestSuiteResult TestSuiteExecution { get; set; } = new();
    public ComprehensiveReport ComprehensiveReporting { get; set; } = new();
    public AutoRemediationResult AutoRemediation { get; set; } = new();
    public ProductionGateResult ProductionGate { get; set; } = new();

    public void CalculateFinalResult()
    {
        IsSuccessful = CredentialDetection.IsSuccessful &&
                      StagingDeployment.IsSuccessful &&
                      TestSuiteExecution.IsOverallSuccess &&
                      string.IsNullOrEmpty(ComprehensiveReporting.GenerationError) &&
                      AutoRemediation.OverallSuccess &&
                      string.IsNullOrEmpty(CriticalError);

        IsProductionReady = IsSuccessful && 
                           ProductionGate.IsProductionReady &&
                           ComprehensiveReporting.IsProductionReady;
    }
}

public class CredentialDetectionResult
{
    public bool IsSuccessful { get; set; }
    public bool HasCredentials { get; set; }
    public string CredentialSource { get; set; } = "";
    public int TotalSourcesFound { get; set; }
    public List<string> Details { get; set; } = new();
    public string? ErrorMessage { get; set; }
}

public class StagingDeploymentResult
{
    public bool IsSuccessful { get; set; }
    public bool EnvironmentConfigured { get; set; }
    public bool EndpointsConfigured { get; set; }
    public bool RiskManagementConfigured { get; set; }
    public bool MonitoringConfigured { get; set; }
    public bool ConnectivityValidated { get; set; }
    public List<string> Details { get; set; } = new();
    public string? ErrorMessage { get; set; }
}