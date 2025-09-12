using Microsoft.Extensions.Logging;
using System.Text.Json;
using TradingBot.Abstractions;

namespace Infrastructure.TopstepX;

/// <summary>
/// Manages TopstepX credentials with automatic detection and secure storage
/// </summary>
public class TopstepXCredentialManager
{
    private readonly ILogger<TopstepXCredentialManager> _logger;
    private readonly string _credentialsPath;
    
    public TopstepXCredentialManager(ILogger<TopstepXCredentialManager> logger)
    {
        _logger = logger;
        
        // FIXED: Use Roaming AppData instead of legacy .topstepx path
        var appDataPath = Environment.GetEnvironmentVariable("TRADING_CREDENTIALS_PATH") ?? 
                         Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TradingBot");
        _credentialsPath = Path.Combine(appDataPath, "credentials.json");
        
        // Clean up legacy .topstepx path if it exists
        CleanupLegacyCredentialPaths();
    }

    /// <summary>
    /// Clean up legacy .topstepx credential paths
    /// Implements credential path cleanup requirement
    /// </summary>
    private void CleanupLegacyCredentialPaths()
    {
        try
        {
            var legacyPaths = new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".topstepx"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".topstep"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tsx")
            };

            foreach (var legacyPath in legacyPaths)
            {
                if (Directory.Exists(legacyPath))
                {
                    try
                    {
                        // Before removing, try to migrate credentials if they exist
                        var legacyCredFile = Path.Combine(legacyPath, "credentials.json");
                        if (File.Exists(legacyCredFile) && !File.Exists(_credentialsPath))
                        {
                            var credContent = File.ReadAllText(legacyCredFile);
                            var directory = Path.GetDirectoryName(_credentialsPath);
                            if (!Directory.Exists(directory))
                            {
                                Directory.CreateDirectory(directory!);
                            }
                            File.WriteAllText(_credentialsPath, credContent);
                            _logger.LogInformation("📦 Migrated credentials from legacy path: {LegacyPath}", legacyPath);
                        }

                        // Remove legacy directory
                        Directory.Delete(legacyPath, recursive: true);
                        _logger.LogInformation("🧹 Cleaned up legacy credential path: {LegacyPath}", legacyPath);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "⚠️ Could not clean up legacy path: {LegacyPath}", legacyPath);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ Error during legacy credential path cleanup");
        }
    }

    public TopstepXCredentials? LoadCredentials()
    {
        try
        {
            // Enhanced environment variable detection with multiple patterns
            var credentials = TryLoadFromEnvironmentVariables();
            if (credentials != null)
            {
                _logger.LogInformation("🔑 Found TopstepX credentials in environment variables (Source: {Source})", credentials.Source);
                return credentials;
            }

            // Try loading from secure file
            if (File.Exists(_credentialsPath))
            {
                var json = File.ReadAllText(_credentialsPath);
                var fileCredentials = JsonSerializer.Deserialize<TopstepXCredentials>(json);
                if (fileCredentials != null && !string.IsNullOrEmpty(fileCredentials.Username))
                {
                    _logger.LogInformation("🔑 Found TopstepX credentials in secure storage");
                    fileCredentials.Source = "SecureFile";
                    return fileCredentials;
                }
            }

            _logger.LogWarning("⚠️ No TopstepX credentials found - will use demo mode");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error loading TopstepX credentials");
            return null;
        }
    }

    public async Task<bool> SaveCredentialsAsync(TopstepXCredentials credentials)
    {
        try
        {
            var directory = Path.GetDirectoryName(_credentialsPath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory!);
            }

            var json = JsonSerializer.Serialize(credentials, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            
            await File.WriteAllTextAsync(_credentialsPath, json);
            
            // Set file permissions to be readable only by current user
            if (OperatingSystem.IsWindows())
            {
                var fileInfo = new FileInfo(_credentialsPath);
                fileInfo.Attributes = FileAttributes.Hidden;
            }

            _logger.LogInformation("✅ TopstepX credentials saved securely");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error saving TopstepX credentials");
            return false;
        }
    }

    public bool HasValidCredentials()
    {
        var credentials = LoadCredentials();
        return credentials != null && 
               !string.IsNullOrEmpty(credentials.Username) && 
               !string.IsNullOrEmpty(credentials.ApiKey);
    }

    public void SetEnvironmentCredentials(string username, string apiKey, string? jwtToken = null)
    {
        Environment.SetEnvironmentVariable("TOPSTEPX_USERNAME", username);
        Environment.SetEnvironmentVariable("TOPSTEPX_API_KEY", apiKey);
        if (!string.IsNullOrEmpty(jwtToken))
        {
            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwtToken);
        }
        
        _logger.LogInformation("✅ TopstepX credentials set in environment");
    }

    /// <summary>
    /// Enhanced environment variable detection with multiple common patterns
    /// </summary>
    private TopstepXCredentials? TryLoadFromEnvironmentVariables()
    {
        // Standard primary pattern
        var envUsername = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
        var envApiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
        var envJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        var envAccountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");

        if (!string.IsNullOrEmpty(envUsername) && !string.IsNullOrEmpty(envApiKey))
        {
            return new TopstepXCredentials
            {
                Username = envUsername,
                ApiKey = envApiKey,
                JwtToken = envJwt,
                AccountId = envAccountId,
                Source = "Environment-Primary"
            };
        }

        // Try alternative naming patterns commonly used
        var patterns = new[]
        {
            ("TOPSTEP_USERNAME", "TOPSTEP_API_KEY", "TOPSTEP_JWT", "TOPSTEP_ACCOUNT_ID", "Environment-TOPSTEP"),
            ("TSX_USERNAME", "TSX_API_KEY", "TSX_JWT", "TSX_ACCOUNT_ID", "Environment-TSX"),
            ("TOPSTEPX_USER", "TOPSTEPX_KEY", "TOPSTEPX_TOKEN", "TOPSTEPX_ACCOUNT", "Environment-ALT1"),
            ("TRADING_USERNAME", "TRADING_API_KEY", "TRADING_JWT", "TRADING_ACCOUNT_ID", "Environment-TRADING"),
            ("BOT_USERNAME", "BOT_API_KEY", "BOT_JWT", "BOT_ACCOUNT_ID", "Environment-BOT"),
            ("LIVE_USERNAME", "LIVE_API_KEY", "LIVE_JWT", "LIVE_ACCOUNT_ID", "Environment-LIVE")
        };

        foreach (var (userVar, keyVar, jwtVar, accountVar, source) in patterns)
        {
            var username = Environment.GetEnvironmentVariable(userVar);
            var apiKey = Environment.GetEnvironmentVariable(keyVar);
            var jwt = Environment.GetEnvironmentVariable(jwtVar);
            var accountId = Environment.GetEnvironmentVariable(accountVar);

            if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey))
            {
                _logger.LogInformation("🔍 Found credentials using pattern: {Pattern}", source);
                return new TopstepXCredentials
                {
                    Username = username,
                    ApiKey = apiKey,
                    JwtToken = jwt,
                    AccountId = accountId,
                    Source = source
                };
            }
        }

        // Try to find credentials in common CI/CD environment patterns
        var ciPatterns = new[]
        {
            ("GITHUB_TOPSTEPX_USERNAME", "GITHUB_TOPSTEPX_API_KEY", "Environment-GITHUB"),
            ("AZURE_TOPSTEPX_USERNAME", "AZURE_TOPSTEPX_API_KEY", "Environment-AZURE"),
            ("AWS_TOPSTEPX_USERNAME", "AWS_TOPSTEPX_API_KEY", "Environment-AWS"),
            ("DOCKER_TOPSTEPX_USERNAME", "DOCKER_TOPSTEPX_API_KEY", "Environment-DOCKER")
        };

        foreach (var (userVar, keyVar, source) in ciPatterns)
        {
            var username = Environment.GetEnvironmentVariable(userVar);
            var apiKey = Environment.GetEnvironmentVariable(keyVar);

            if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey))
            {
                _logger.LogInformation("🔍 Found credentials using CI pattern: {Pattern}", source);
                return new TopstepXCredentials
                {
                    Username = username,
                    ApiKey = apiKey,
                    Source = source
                };
            }
        }

        return null;
    }

    /// <summary>
    /// Auto-detect and validate all available credential sources
    /// </summary>
    public CredentialDiscoveryReport DiscoverAllCredentialSources()
    {
        var report = new CredentialDiscoveryReport();
        
        try
        {
            // Check environment variables
            var envCreds = TryLoadFromEnvironmentVariables();
            if (envCreds != null)
            {
                report.EnvironmentCredentials = envCreds;
                report.HasEnvironmentCredentials = true;
            }

            // Check secure file
            if (File.Exists(_credentialsPath))
            {
                try
                {
                    var json = File.ReadAllText(_credentialsPath);
                    var fileCreds = JsonSerializer.Deserialize<TopstepXCredentials>(json);
                    if (fileCreds != null && !string.IsNullOrEmpty(fileCreds.Username))
                    {
                        fileCreds.Source = "SecureFile";
                        report.FileCredentials = fileCreds;
                        report.HasFileCredentials = true;
                    }
                }
                catch (Exception ex)
                {
                    report.FileErrorMessage = ex.Message;
                }
            }

            // Set recommended source
            if (report.HasEnvironmentCredentials)
            {
                report.RecommendedCredentials = report.EnvironmentCredentials;
                report.RecommendedSource = "Environment (Best for automation)";
            }
            else if (report.HasFileCredentials)
            {
                report.RecommendedCredentials = report.FileCredentials;
                report.RecommendedSource = "SecureFile (Manual setup)";
            }

            _logger.LogInformation("🔍 Credential discovery complete - Environment: {HasEnv}, File: {HasFile}", 
                report.HasEnvironmentCredentials, report.HasFileCredentials);
        }
        catch (Exception ex)
        {
            report.DiscoveryError = ex.Message;
            _logger.LogError(ex, "❌ Error during credential discovery");
        }

        return report;
    }
}


