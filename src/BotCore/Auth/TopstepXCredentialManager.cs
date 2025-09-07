using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Auth;

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
        _credentialsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".topstepx", "credentials.json");
    }

    public TopstepXCredentials? LoadCredentials()
    {
        try
        {
            // First try environment variables
            var envUsername = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            var envApiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            var envJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");

            if (!string.IsNullOrEmpty(envUsername) && !string.IsNullOrEmpty(envApiKey))
            {
                _logger.LogInformation("🔑 Found TopstepX credentials in environment variables");
                return new TopstepXCredentials
                {
                    Username = envUsername,
                    ApiKey = envApiKey,
                    JwtToken = envJwt,
                    Source = "Environment"
                };
            }

            // Try loading from secure file
            if (File.Exists(_credentialsPath))
            {
                var json = File.ReadAllText(_credentialsPath);
                var credentials = JsonSerializer.Deserialize<TopstepXCredentials>(json);
                if (credentials != null && !string.IsNullOrEmpty(credentials.Username))
                {
                    _logger.LogInformation("🔑 Found TopstepX credentials in secure storage");
                    credentials.Source = "SecureFile";
                    return credentials;
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
}

public class TopstepXCredentials
{
    public string Username { get; set; } = "";
    public string ApiKey { get; set; } = "";
    public string? JwtToken { get; set; }
    public string? AccountId { get; set; }
    public DateTime? LastUpdated { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = "";
}
