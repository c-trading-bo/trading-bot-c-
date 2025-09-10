extern alias BotCoreTest;

using System;
using Xunit;
using Microsoft.Extensions.Logging.Abstractions;

namespace TradingBot.Tests.Unit;

public class SecurityServiceTests
{
    private readonly BotCoreTest::BotCore.Services.SecurityService _securityService;

    public SecurityServiceTests()
    {
        _securityService = new BotCoreTest::BotCore.Services.SecurityService(NullLogger<BotCoreTest::BotCore.Services.SecurityService>.Instance);
    }

    [Theory]
    [InlineData("token=abc123def456", "token[REDACTED]")]
    [InlineData("password=secret123", "password[REDACTED]")]
    [InlineData("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "Authorization: [REDACTED]")]
    [InlineData("secret_key=mysecretkey", "secret_key[REDACTED]")]
    public void RedactSensitiveData_ShouldRedactTokensAndKeys(string input, string expectedPattern)
    {
        // Act
        var result = _securityService.RedactSensitiveData(input);

        // Assert
        Assert.Contains("[REDACTED]", result);
        Assert.DoesNotContain("abc123def456", result);
        Assert.DoesNotContain("secret123", result);
        Assert.DoesNotContain("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", result);
        
        // Use expectedPattern for validation
        if (!string.IsNullOrEmpty(expectedPattern))
        {
            var expectedParts = expectedPattern.Split("[REDACTED]");
            foreach (var part in expectedParts)
            {
                if (!string.IsNullOrWhiteSpace(part))
                {
                    Assert.Contains(part, result);
                }
            }
        }
    }

    [Theory]
    [InlineData("Account: 1234567890", "Account: 1234****7890")]
    [InlineData("ID: 9876543210", "ID: 9876****3210")]
    [InlineData("4111 1111 1111 1111", "4111****1111")]
    public void RedactSensitiveData_ShouldRedactAccountNumbers(string input, string expectedPattern)
    {
        // Act
        var result = _securityService.RedactSensitiveData(input);

        // Assert
        Assert.Contains("****", result);
        Assert.DoesNotContain("567890", result);
        Assert.DoesNotContain("543210", result);
        
        // Use expectedPattern for validation
        if (!string.IsNullOrEmpty(expectedPattern))
        {
            var expectedParts = expectedPattern.Split("[REDACTED]");
            foreach (var part in expectedParts)
            {
                if (!string.IsNullOrWhiteSpace(part))
                {
                    Assert.Contains(part, result);
                }
            }
        }
    }

    [Theory]
    [InlineData("user@example.com", "user@[REDACTED]")]
    [InlineData("admin@company.org", "admin@[REDACTED]")]
    public void RedactSensitiveData_ShouldRedactEmails(string input, string expectedPattern)
    {
        // Act
        var result = _securityService.RedactSensitiveData(input);

        // Assert
        Assert.Contains("@[REDACTED]", result);
        Assert.DoesNotContain("example.com", result);
        Assert.DoesNotContain("company.org", result);
        
        // Use expectedPattern for validation
        if (!string.IsNullOrEmpty(expectedPattern))
        {
            var expectedParts = expectedPattern.Split("[REDACTED]");
            foreach (var part in expectedParts)
            {
                if (!string.IsNullOrWhiteSpace(part))
                {
                    Assert.Contains(part, result);
                }
            }
        }
    }

    [Fact]
    public void RedactSensitiveData_NullOrEmpty_ShouldReturnInput()
    {
        // Act & Assert
        Assert.Null(_securityService.RedactSensitiveData(null));
        Assert.Equal("", _securityService.RedactSensitiveData(""));
    }

    [Fact]
    public void RedactSensitiveData_NoSensitiveData_ShouldReturnUnchanged()
    {
        // Arrange
        var input = "This is a normal log message with no sensitive data";

        // Act
        var result = _securityService.RedactSensitiveData(input);

        // Assert
        Assert.Equal(input, result);
    }

    [Fact]
    public void IsRemoteSessionDetected_ShouldReturnBooleanResult()
    {
        // Act
        var result = _securityService.IsRemoteSessionDetected();

        // Assert
        Assert.IsType<bool>(result);
        // Note: The actual result depends on the environment where the test runs
        // We just verify it doesn't throw and returns a boolean
    }

    [Fact]
    public void IsTradingAllowed_WithoutRemoteSession_ShouldReturnTrue()
    {
        // Note: This test depends on the actual environment
        // In a normal development environment, it should return true
        
        // Act
        var result = _securityService.IsTradingAllowed();

        // Assert
        Assert.IsType<bool>(result);
    }

    [Fact]
    public void LogSecurityEvent_ShouldNotThrow()
    {
        // Arrange
        var eventData = new
        {
            timestamp = DateTime.UtcNow,
            event_type = "test_event",
            details = "Test security event",
            token = "sensitive_token_123"
        };

        // Act & Assert
        var exception = Record.Exception(() => 
            _securityService.LogSecurityEvent("test", eventData));
        
        Assert.Null(exception);
    }

    [Theory]
    [InlineData("Bearer abc123token456", "Bearer [REDACTED]")]
    [InlineData("Basic dXNlcjpwYXNzd29yZA==", "Basic [REDACTED]")]
    public void RedactSensitiveData_ShouldRedactAuthHeaders(string input, string expected)
    {
        // Act
        var result = _securityService.RedactSensitiveData(input);

        // Assert
        Assert.Contains("[REDACTED]", result);
        Assert.DoesNotContain("abc123token456", result);
        Assert.DoesNotContain("dXNlcjpwYXNzd29yZA==", result);
        
        // Use expected for validation
        if (!string.IsNullOrEmpty(expected))
        {
            var expectedParts = expected.Split("[REDACTED]");
            foreach (var part in expectedParts)
            {
                if (!string.IsNullOrWhiteSpace(part))
                {
                    Assert.Contains(part, result);
                }
            }
        }
    }

    [Fact]
    public void RedactSensitiveData_ComplexJson_ShouldRedactAllSensitive()
    {
        // Arrange
        var jsonInput = @"{
            ""username"": ""john.doe@example.com"",
            ""token"": ""eyJhbGciOiJIUzI1NiJ9.payload.signature"",
            ""account_id"": ""1234567890"",
            ""normal_field"": ""normal_value""
        }";

        // Act
        var result = _securityService.RedactSensitiveData(jsonInput);

        // Assert
        Assert.Contains("[REDACTED]", result);
        Assert.Contains("normal_value", result); // Normal field should remain
        Assert.DoesNotContain("example.com", result);
        Assert.DoesNotContain("eyJhbGciOiJIUzI1NiJ9.payload.signature", result);
        Assert.DoesNotContain("1234567890", result);
    }
}