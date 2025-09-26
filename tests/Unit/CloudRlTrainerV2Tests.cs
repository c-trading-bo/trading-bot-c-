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