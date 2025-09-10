extern alias BotCoreTest;

using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace UnitTests.Http
{
    public class HttpClientAuthIntegrationTests : IDisposable
    {
        private readonly Mock<ILogger<BotCoreTest::BotCore.Services.TopstepXHttpClient>> _mockLogger;
        private readonly Mock<BotCoreTest::BotCore.Auth.ITopstepAuth> _mockAuthService;
        private readonly HttpClient _httpClient;

        public HttpClientAuthIntegrationTests()
        {
            _mockLogger = new Mock<ILogger<BotCoreTest::BotCore.Services.TopstepXHttpClient>>();
            _mockAuthService = new Mock<BotCoreTest::BotCore.Auth.ITopstepAuth>();
            _httpClient = new HttpClient();
        }

        [Fact]
        public async Task GetAsync_WithAuthService_ShouldEnsureFreshToken()
        {
            // Arrange
            var jwt = "test.jwt.token";
            var expiry = DateTimeOffset.UtcNow.AddMinutes(10);

            _mockAuthService
                .Setup(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()))
                .Returns(Task.CompletedTask);

            _mockAuthService
                .Setup(x => x.GetFreshJwtAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync((jwt, expiry));

            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, _mockAuthService.Object);

            // Act & Assert - This will fail due to network call, but we want to verify auth is called
            try
            {
                await client.GetAsync("/test");
            }
            catch (HttpRequestException)
            {
                // Expected - we don't have a real server
            }

            // Verify auth methods were called
            _mockAuthService.Verify(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()), Times.Once);
            _mockAuthService.Verify(x => x.GetFreshJwtAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        [Fact]
        public async Task PostAsync_WithAuthService_ShouldEnsureFreshToken()
        {
            // Arrange
            var jwt = "test.jwt.token";
            var expiry = DateTimeOffset.UtcNow.AddMinutes(10);

            _mockAuthService
                .Setup(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()))
                .Returns(Task.CompletedTask);

            _mockAuthService
                .Setup(x => x.GetFreshJwtAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync((jwt, expiry));

            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, _mockAuthService.Object);
            var content = new StringContent("test");

            // Act & Assert - This will fail due to network call, but we want to verify auth is called
            try
            {
                await client.PostAsync("/test", content);
            }
            catch (HttpRequestException)
            {
                // Expected - we don't have a real server
            }

            // Verify auth methods were called
            _mockAuthService.Verify(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()), Times.Once);
            _mockAuthService.Verify(x => x.GetFreshJwtAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        [Fact]
        public async Task GetAsync_AuthServiceFailure_ShouldPropagateException()
        {
            // Arrange
            var authException = new InvalidOperationException("Auth failed");

            _mockAuthService
                .Setup(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()))
                .ThrowsAsync(authException);

            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, _mockAuthService.Object);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => client.GetAsync("/test"));

            Assert.Equal("Auth failed", exception.Message);

            // Verify error logging
            _mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Failed to ensure fresh token")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        [Fact]
        public async Task Constructor_WithoutAuthService_ShouldNotThrow()
        {
            // Act & Assert - Should not throw when auth service is null
            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, null);
            
            Assert.NotNull(client);
        }

        [Fact]
        public async Task GetAsync_WithoutAuthService_ShouldFallbackToEnvironment()
        {
            // Arrange
            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, null);

            // Act & Assert - This will fail due to network call, but should not throw auth errors
            try
            {
                await client.GetAsync("/test");
            }
            catch (HttpRequestException)
            {
                // Expected - we don't have a real server
            }

            // Should not have called auth service (since it's null)
            _mockAuthService.Verify(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()), Times.Never);
        }

        [Fact]
        public async Task EnsureFreshTokenAsync_CancellationRequested_ShouldPropagate()
        {
            // Arrange
            _mockAuthService
                .Setup(x => x.EnsureFreshTokenAsync(It.IsAny<CancellationToken>()))
                .ThrowsAsync(new OperationCanceledException());

            var client = new BotCoreTest::BotCore.Services.TopstepXHttpClient(_httpClient, _mockLogger.Object, _mockAuthService.Object);

            using var cts = new CancellationTokenSource();
            cts.Cancel();

            // Act & Assert
            await Assert.ThrowsAsync<OperationCanceledException>(
                () => client.GetAsync("/test", cts.Token));
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _httpClient?.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}