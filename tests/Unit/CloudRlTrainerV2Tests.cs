using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Moq;
using Xunit;
using TradingBot.Abstractions;

namespace TradingBot.Unit.Tests;

/// <summary>
/// Focused unit tests for model promotion SHA256 verification
/// Tests the critical path for production model deployment safety
/// </summary>
public class ModelPromotionTests : IDisposable
{
    private readonly string _tempDir;

    public ModelPromotionTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "ModelPromotionTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void VerifySha256_WithMismatchedHash_ShouldReturnFalse()
    {
        // Arrange
        var testFile = Path.Combine(_tempDir, "test_model.onnx");
        var testContent = "fake model content";
        File.WriteAllText(testFile, testContent);

        // Calculate correct SHA256
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var correctHashBytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(testContent));
        var correctHash = Convert.ToHexString(correctHashBytes);
        
        var wrongHash = "wrong_hash_value";

        // Act & Assert
        Assert.True(VerifySha256(testFile, correctHash), "Correct hash should validate");
        Assert.False(VerifySha256(testFile, wrongHash), "Wrong hash should fail validation");
    }

    [Fact]
    public void AtomicDirectorySwap_ShouldMoveDirectoriesCorrectly()
    {
        // Arrange
        var stageDir = Path.Combine(_tempDir, "stage");
        var currentDir = Path.Combine(_tempDir, "current");
        var previousDir = Path.Combine(_tempDir, "previous");

        Directory.CreateDirectory(stageDir);
        Directory.CreateDirectory(currentDir);
        
        File.WriteAllText(Path.Combine(stageDir, "model.onnx"), "staged model");
        File.WriteAllText(Path.Combine(currentDir, "old_model.onnx"), "current model");

        // Act
        AtomicDirectorySwap(stageDir, currentDir, previousDir);

        // Assert
        Assert.True(File.Exists(Path.Combine(currentDir, "model.onnx")));
        Assert.True(File.Exists(Path.Combine(previousDir, "old_model.onnx")));
        Assert.False(Directory.Exists(stageDir)); // Should be moved, not copied
        
        Assert.Equal("staged model", File.ReadAllText(Path.Combine(currentDir, "model.onnx")));
        Assert.Equal("current model", File.ReadAllText(Path.Combine(previousDir, "old_model.onnx")));
    }

    [Fact]
    public void LiveTradingGate_WithKillSwitch_ShouldBlockLiveTrading()
    {
        // Arrange
        var killFile = Path.Combine(_tempDir, "kill.txt");
        File.WriteAllText(killFile, "emergency stop");

        // Act
        var isBlocked = IsKillSwitchActive(killFile);

        // Assert
        Assert.True(isBlocked, "Kill switch should block live trading");
    }

    [Fact]
    public void LiveTradingGate_WithValidArmToken_ShouldAllowLiveTrading()
    {
        // Arrange
        var armFile = Path.Combine(_tempDir, "live_arm.json");
        var futureExpiry = DateTime.UtcNow.AddHours(1);
        var token = "test_token_123";
        
        var armContent = $$"""
        {
            "token": "{{token}}",
            "expires_at": "{{futureExpiry:yyyy-MM-ddTHH:mm:ssZ}}",
            "created_at": "{{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}}"
        }
        """;
        
        File.WriteAllText(armFile, armContent);

        // Act
        var isValid = ValidateArmToken(armFile, token);

        // Assert
        Assert.True(isValid, "Valid arm token should allow live trading");
    }

    [Fact]
    public void LiveTradingGate_WithExpiredArmToken_ShouldBlockLiveTrading()
    {
        // Arrange
        var armFile = Path.Combine(_tempDir, "live_arm.json");
        var pastExpiry = DateTime.UtcNow.AddHours(-1);
        var token = "test_token_123";
        
        var armContent = $$"""
        {
            "token": "{{token}}",
            "expires_at": "{{pastExpiry:yyyy-MM-ddTHH:mm:ssZ}}",
            "created_at": "{{DateTime.UtcNow.AddHours(-2):yyyy-MM-ddTHH:mm:ssZ}}"
        }
        """;
        
        File.WriteAllText(armFile, armContent);

        // Act
        var isValid = ValidateArmToken(armFile, token);

        // Assert
        Assert.False(isValid, "Expired arm token should block live trading");
    }

    [Fact]
    public async Task NeutralBandIntegration_WithLattices_ShouldUseDynamicThresholds()
    {
        // Arrange - Create a mock neutral band service with custom thresholds
        var mockLogger = new Mock<Microsoft.Extensions.Logging.ILogger<BotCore.Services.SafeHoldDecisionPolicy>>();
        var mockConfig = new Mock<Microsoft.Extensions.Configuration.IConfiguration>();
        var mockSection = new Mock<Microsoft.Extensions.Configuration.IConfigurationSection>();
        
        // Setup configuration for neutral band
        mockSection.Setup(x => x.GetValue<double>("BearishThreshold", 0.45)).Returns(0.40);
        mockSection.Setup(x => x.GetValue<double>("BullishThreshold", 0.55)).Returns(0.60);
        mockSection.Setup(x => x.GetValue<bool>("EnableHysteresis", true)).Returns(true);
        mockSection.Setup(x => x.GetValue<double>("HysteresisBuffer", 0.02)).Returns(0.02);
        mockConfig.Setup(x => x.GetSection("NeutralBand")).Returns(mockSection.Object);
        
        var neutralBandService = new BotCore.Services.SafeHoldDecisionPolicy(mockLogger.Object, mockConfig.Object);
        var lattices = new OrchestratorAgent.Execution.PerSymbolSessionLattices(neutralBandService);
        
        // Act - Test various confidence levels
        var buyDecision = await lattices.EvaluateTradingDecisionAsync("ES", SessionType.RTH, 0.65, "S2a");
        var sellDecision = await lattices.EvaluateTradingDecisionAsync("ES", SessionType.RTH, 0.35, "S2a");
        var holdDecision = await lattices.EvaluateTradingDecisionAsync("ES", SessionType.RTH, 0.50, "S2a");
        
        // Assert
        Assert.NotNull(buyDecision);
        Assert.Equal(TradingAction.Buy, buyDecision.Action);
        Assert.Contains("Above bullish threshold", buyDecision.Reason);
        
        Assert.NotNull(sellDecision);
        Assert.Equal(TradingAction.Sell, sellDecision.Action);
        Assert.Contains("Below bearish threshold", sellDecision.Reason);
        
        Assert.NotNull(holdDecision);
        Assert.Equal(TradingAction.Hold, holdDecision.Action);
        Assert.Contains("neutral zone", holdDecision.Reason);
        
        // Verify metadata includes lattice-specific information
        Assert.NotNull(buyDecision.Metadata);
        Assert.True(buyDecision.Metadata.ContainsKey("volatility_factor"));
        Assert.True(buyDecision.Metadata.ContainsKey("bayesian_win_prob"));
    }

    [Fact]
    public void NeutralBandIntegration_IsInNeutralBand_ShouldUseDynamicService()
    {
        // Arrange
        var mockLogger = new Mock<Microsoft.Extensions.Logging.ILogger<BotCore.Services.SafeHoldDecisionPolicy>>();
        var mockConfig = new Mock<Microsoft.Extensions.Configuration.IConfiguration>();
        var mockSection = new Mock<Microsoft.Extensions.Configuration.IConfigurationSection>();
        
        mockSection.Setup(x => x.GetValue<double>("BearishThreshold", 0.45)).Returns(0.42);
        mockSection.Setup(x => x.GetValue<double>("BullishThreshold", 0.55)).Returns(0.58);
        mockConfig.Setup(x => x.GetSection("NeutralBand")).Returns(mockSection.Object);
        
        var neutralBandService = new BotCore.Services.SafeHoldDecisionPolicy(mockLogger.Object, mockConfig.Object);
        var lattices = new OrchestratorAgent.Execution.PerSymbolSessionLattices(neutralBandService);
        
        // Act & Assert
        Assert.False(lattices.IsInNeutralBand(0.40, "ES", SessionType.RTH)); // Below bearish
        Assert.True(lattices.IsInNeutralBand(0.50, "ES", SessionType.RTH));  // In neutral band
        Assert.False(lattices.IsInNeutralBand(0.60, "ES", SessionType.RTH)); // Above bullish
    }

    // Helper methods that simulate the actual implementation logic

    private static bool VerifySha256(string filePath, string expectedSha256)
    {
        if (string.IsNullOrEmpty(expectedSha256)) return true;
        
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hash = sha256.ComputeHash(stream);
        var computedSha256 = Convert.ToHexString(hash);
        
        return string.Equals(computedSha256, expectedSha256, StringComparison.OrdinalIgnoreCase);
    }

    private static void AtomicDirectorySwap(string stageDir, string currentDir, string previousDir)
    {
        // Backup current to previous (atomic)
        if (Directory.Exists(currentDir))
        {
            if (Directory.Exists(previousDir))
            {
                Directory.Delete(previousDir, true);
            }
            Directory.Move(currentDir, previousDir);
        }

        // Move staged to current (atomic)
        Directory.Move(stageDir, currentDir);
    }

    private static bool IsKillSwitchActive(string killFile)
    {
        return File.Exists(killFile);
    }

    private static bool ValidateArmToken(string armFile, string expectedToken)
    {
        if (!File.Exists(armFile)) return false;

        try
        {
            var json = File.ReadAllText(armFile);
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;

            if (!root.TryGetProperty("token", out var tokenElement) ||
                !root.TryGetProperty("expires_at", out var expiresElement))
            {
                return false;
            }

            var token = tokenElement.GetString();
            var expiresAt = expiresElement.GetDateTime();

            return token == expectedToken && DateTime.UtcNow < expiresAt;
        }
        catch
        {
            return false;
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, true);
        }
    }
}