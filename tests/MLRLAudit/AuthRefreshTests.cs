extern alias BotCoreTest;

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;
using Infrastructure.TopstepX;
using TradingBot.Abstractions;

namespace UnitTests.Auth
{
    public class AuthRefreshTests
    {
        private readonly Mock<ILogger<CachedTopstepAuth>> _mockLogger;
        
        public AuthRefreshTests()
        {
            _mockLogger = new Mock<ILogger<CachedTopstepAuth>>();
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_WithinRefreshWindow_ShouldNotRefresh()
        {
            // Arrange
            var refreshCallCount = 0;
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(10); // Token expires in 10 minutes
            
            var mockJwt = CreateMockJwt(futureExpiry);
            
            Func<CancellationToken, Task<string>> fetchJwt = _ =>
            {
                refreshCallCount++;
                return Task.FromResult(mockJwt);
            };

            var auth = new CachedTopstepAuth(fetchJwt, _mockLogger.Object);
            
            // Initial fetch to set token
            await auth.GetFreshJwtAsync();
            
            // Reset counter after initial fetch
            refreshCallCount = 0;

            // Act
            await auth.EnsureFreshTokenAsync();

            // Assert
            Assert.Equal(0, refreshCallCount); // Should not refresh as token is still valid
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_OutsideRefreshWindow_ShouldRefresh()
        {
            // Arrange
            var refreshCallCount = 0;
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(3); // Token expires in 3 minutes (< 5 minute window)
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(15); // New token expires in 15 minutes
            
            var initialJwt = CreateMockJwt(nearExpiry);
            var refreshedJwt = CreateMockJwt(futureExpiry);
            
            Func<CancellationToken, Task<string>> fetchJwt = _ =>
            {
                refreshCallCount++;
                return Task.FromResult(refreshCallCount == 1 ? initialJwt : refreshedJwt);
            };

            var auth = new CachedTopstepAuth(fetchJwt, _mockLogger.Object);
            
            // Initial fetch to set expiring token
            await auth.GetFreshJwtAsync();
            
            // Reset counter after initial fetch
            refreshCallCount = 0;

            // Act
            await auth.EnsureFreshTokenAsync();

            // Assert
            Assert.Equal(1, refreshCallCount); // Should refresh once
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_ConcurrentCalls_ShouldRefreshOnlyOnce()
        {
            // Arrange
            var refreshCallCount = 0;
            var refreshDelay = TimeSpan.FromMilliseconds(100);
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(3);
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(15);
            
            var initialJwt = CreateMockJwt(nearExpiry);
            var refreshedJwt = CreateMockJwt(futureExpiry);
            
            Func<CancellationToken, Task<string>> fetchJwt = async _ =>
            {
                Interlocked.Increment(ref refreshCallCount);
                await Task.Delay(refreshDelay); // Simulate network delay
                return refreshCallCount == 1 ? initialJwt : refreshedJwt;
            };

            var auth = new CachedTopstepAuth(fetchJwt, _mockLogger.Object);
            
            // Initial fetch to set expiring token
            await auth.GetFreshJwtAsync();
            
            // Reset counter after initial fetch
            refreshCallCount = 0;

            // Act - Start multiple concurrent refresh attempts
            var tasks = new[]
            {
                auth.EnsureFreshTokenAsync(),
                auth.EnsureFreshTokenAsync(),
                auth.EnsureFreshTokenAsync(),
                auth.EnsureFreshTokenAsync(),
                auth.EnsureFreshTokenAsync()
            };

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(1, refreshCallCount); // Should refresh only once despite concurrent calls
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_RefreshFailure_ShouldThrowAndLog()
        {
            // Arrange
            var refreshException = new InvalidOperationException("Refresh failed");
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(3);
            var initialJwt = CreateMockJwt(nearExpiry);
            
            var callCount = 0;
            Func<CancellationToken, Task<string>> fetchJwt = _ =>
            {
                callCount++;
                if (callCount == 1)
                    return Task.FromResult(initialJwt);
                throw refreshException;
            };

            var auth = new CachedTopstepAuth(fetchJwt, _mockLogger.Object);
            
            // Initial fetch to set expiring token
            await auth.GetFreshJwtAsync();

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => auth.EnsureFreshTokenAsync());
            
            Assert.Equal("Refresh failed", exception.Message);
            
            // Verify error logging
            _mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("JWT refresh failed")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_CancellationRequested_ShouldThrow()
        {
            // Arrange
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(3);
            var initialJwt = CreateMockJwt(nearExpiry);
            
            var callCount = 0;
            Func<CancellationToken, Task<string>> fetchJwt = async ct =>
            {
                callCount++;
                if (callCount == 1)
                    return initialJwt;
                
                await Task.Delay(1000, ct); // Long delay to trigger cancellation
                return CreateMockJwt(DateTimeOffset.UtcNow.AddMinutes(15));
            };

            var auth = new CachedTopstepAuth(fetchJwt, _mockLogger.Object);
            
            // Initial fetch to set expiring token
            await auth.GetFreshJwtAsync();

            using var cts = new CancellationTokenSource();
            cts.Cancel(); // Cancel immediately

            // Act & Assert
            await Assert.ThrowsAsync<OperationCanceledException>(
                () => auth.EnsureFreshTokenAsync(cts.Token));
        }

        private static string CreateMockJwt(DateTimeOffset expiry)
        {
            var header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"; // {"alg":"HS256","typ":"JWT"}
            var payload = Convert.ToBase64String(
                System.Text.Encoding.UTF8.GetBytes($"{{\"exp\":{expiry.ToUnixTimeSeconds()}}}"))
                .Replace('+', '-')
                .Replace('/', '_')
                .TrimEnd('=');
            var signature = "test-signature";
            
            return $"{header}.{payload}.{signature}";
        }
    }
}