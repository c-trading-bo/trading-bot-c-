using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;
using Infrastructure.TopstepX;
using TradingBot.Abstractions;

namespace UnitTests.Integration
{
    /// <summary>
    /// Integration tests demonstrating the complete proactive JWT refresh functionality
    /// </summary>
    public class ProactiveJwtRefreshIntegrationTests
    {
        [Fact]
        public async Task CompleteJwtRefreshFlow_ShouldWork()
        {
            // Arrange
            var logger = Mock.Of<ILogger<CachedTopstepAuth>>();
            var refreshCount = 0;
            var initialExpiry = DateTimeOffset.UtcNow.AddMinutes(3); // Needs refresh
            var newExpiry = DateTimeOffset.UtcNow.AddMinutes(20); // Fresh token
            
            Func<CancellationToken, Task<string>> fetchJwt = async ct =>
            {
                await Task.Delay(50, ct); // Simulate network call
                refreshCount++;
                
                var expiry = refreshCount == 1 ? initialExpiry : newExpiry;
                return CreateMockJwt(expiry);
            };

            var auth = new CachedTopstepAuth(fetchJwt, logger);

            // Act - Get initial token
            var (initialJwt, initialExp) = await auth.GetFreshJwtAsync();
            
            // Reset count to track actual refresh
            refreshCount = 0;
            
            // Ensure fresh token should trigger refresh (within 5-minute window)
            await auth.EnsureFreshTokenAsync();
            
            // Get the refreshed token
            var (refreshedJwt, refreshedExp) = await auth.GetFreshJwtAsync();

            // Assert
            Assert.Equal(1, refreshCount); // Should have refreshed once
            Assert.NotEqual(initialJwt, refreshedJwt); // Token should be different
            Assert.True(refreshedExp > initialExp); // New expiry should be later
            Assert.True(refreshedExp > DateTimeOffset.UtcNow.AddMinutes(15)); // Should be well in the future
        }

        [Fact]
        public async Task ProactiveRefresh_Under5MinuteWindow_ShouldRefresh()
        {
            // Arrange
            var logger = Mock.Of<ILogger<CachedTopstepAuth>>();
            var refreshCount = 0;
            
            // Token expires in 4 minutes - within the 5-minute refresh window
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(4);
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(20);
            
            Func<CancellationToken, Task<string>> fetchJwt = ct =>
            {
                refreshCount++;
                var expiry = refreshCount == 1 ? nearExpiry : futureExpiry;
                return Task.FromResult(CreateMockJwt(expiry));
            };

            var auth = new CachedTopstepAuth(fetchJwt, logger);

            // Act - Get initial token
            await auth.GetFreshJwtAsync();
            refreshCount = 0; // Reset after initial fetch

            // This should trigger refresh because we're within 5-minute window
            await auth.EnsureFreshTokenAsync();

            // Assert
            Assert.Equal(1, refreshCount);
        }

        [Fact]
        public async Task ProactiveRefresh_Over5MinuteWindow_ShouldNotRefresh()
        {
            // Arrange
            var logger = Mock.Of<ILogger<CachedTopstepAuth>>();
            var refreshCount = 0;
            
            // Token expires in 6 minutes - outside the 5-minute refresh window
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(6);
            
            Func<CancellationToken, Task<string>> fetchJwt = ct =>
            {
                refreshCount++;
                return Task.FromResult(CreateMockJwt(futureExpiry));
            };

            var auth = new CachedTopstepAuth(fetchJwt, logger);

            // Act - Get initial token
            await auth.GetFreshJwtAsync();
            refreshCount = 0; // Reset after initial fetch

            // This should NOT trigger refresh because we're outside 5-minute window
            await auth.EnsureFreshTokenAsync();

            // Assert
            Assert.Equal(0, refreshCount);
        }

        [Fact]
        public async Task SingleFlightRefresh_ConcurrentCalls_ShouldRefreshOnlyOnce()
        {
            // Arrange
            var logger = Mock.Of<ILogger<CachedTopstepAuth>>();
            var refreshCount = 0;
            var refreshDelay = TimeSpan.FromMilliseconds(200);
            
            var nearExpiry = DateTimeOffset.UtcNow.AddMinutes(3);
            var futureExpiry = DateTimeOffset.UtcNow.AddMinutes(20);
            
            Func<CancellationToken, Task<string>> fetchJwt = async ct =>
            {
                var currentCount = Interlocked.Increment(ref refreshCount);
                await Task.Delay(refreshDelay, ct); // Simulate slow network
                
                var expiry = currentCount == 1 ? nearExpiry : futureExpiry;
                return CreateMockJwt(expiry);
            };

            var auth = new CachedTopstepAuth(fetchJwt, logger);

            // Get initial token
            await auth.GetFreshJwtAsync();
            refreshCount = 0; // Reset after initial fetch

            // Act - Start 10 concurrent refresh attempts
            var tasks = new Task[10];
            for (int i = 0; i < 10; i++)
            {
                tasks[i] = auth.EnsureFreshTokenAsync();
            }

            await Task.WhenAll(tasks);

            // Assert - Only one refresh should have occurred
            Assert.Equal(1, refreshCount);
        }

        [Fact]
        public async Task ErrorHandling_RefreshFailure_ShouldPreserveOldToken()
        {
            // Arrange
            var logger = Mock.Of<ILogger<CachedTopstepAuth>>();
            var callCount = 0;
            var goodExpiry = DateTimeOffset.UtcNow.AddMinutes(3); // Needs refresh
            
            Func<CancellationToken, Task<string>> fetchJwt = ct =>
            {
                callCount++;
                if (callCount == 1)
                {
                    return Task.FromResult(CreateMockJwt(goodExpiry));
                }
                throw new InvalidOperationException("Network error");
            };

            var auth = new CachedTopstepAuth(fetchJwt, logger);

            // Get initial token
            var (initialJwt, _) = await auth.GetFreshJwtAsync();

            // Act & Assert - Refresh should fail but preserve old token
            await Assert.ThrowsAsync<InvalidOperationException>(() => auth.EnsureFreshTokenAsync());
            
            // Old token should still be available
            var (currentJwt, _) = await auth.GetFreshJwtAsync();
            Assert.Equal(initialJwt, currentJwt);
        }

        private static string CreateMockJwt(DateTimeOffset expiry)
        {
            var header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"; // {"alg":"HS256","typ":"JWT"}
            var payload = Convert.ToBase64String(
                System.Text.Encoding.UTF8.GetBytes($"{{\"exp\":{expiry.ToUnixTimeSeconds()}}}"))
                .Replace('+', '-')
                .Replace('/', '_')
                .TrimEnd('=');
            var signature = Convert.ToBase64String(
                System.Text.Encoding.UTF8.GetBytes($"signature-{expiry.ToUnixTimeSeconds()}"))
                .Replace('+', '-')
                .Replace('/', '_')
                .TrimEnd('=');
            
            return $"{header}.{payload}.{signature}";
        }
    }
}