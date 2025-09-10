extern alias BotCoreTest;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Xunit;
using System.Threading.Tasks;
using System.Threading;

namespace UnitTests.DI
{
    public class ServiceRegistrationTests
    {
        [Fact]
        public void AddTopstepAuthentication_ShouldRegisterAllServices()
        {
            // Arrange
            var services = new ServiceCollection();
            services.AddLogging();
            
            var configuration = new ConfigurationBuilder().Build();

            // Act
            services.AddTopstepAuthenticationLegacy(configuration);
            var serviceProvider = services.BuildServiceProvider();

            // Assert
            var httpClient = serviceProvider.GetService<BotCoreTest::BotCore.Services.ITopstepXHttpClient>();
            Assert.NotNull(httpClient);
        }

        [Fact]
        public void AddTopstepAuthentication_WithAuth_ShouldRegisterAuthService()
        {
            // Arrange
            var services = new ServiceCollection();
            services.AddLogging();
            
            var configuration = new ConfigurationBuilder().Build();
            
            var customAuthProvider = (CancellationToken ct) => Task.FromResult("test-jwt");

            // Act
            services.AddTopstepAuthentication(configuration, customAuthProvider);
            var serviceProvider = services.BuildServiceProvider();

            // Assert
            var authService = serviceProvider.GetService<BotCoreTest::BotCore.Auth.ITopstepAuth>();
            var httpClient = serviceProvider.GetService<BotCoreTest::BotCore.Services.ITopstepXHttpClient>();
            
            Assert.NotNull(authService);
            Assert.NotNull(httpClient);
        }

        [Fact]
        public void ServiceRegistration_ShouldUseSingletonInstances()
        {
            // Arrange
            var services = new ServiceCollection();
            services.AddLogging();
            
            var configuration = new ConfigurationBuilder().Build();

            // Act
            services.AddTopstepAuthenticationLegacy(configuration);
            var serviceProvider = services.BuildServiceProvider();

            // Assert - Should return same instance
            var httpClient1 = serviceProvider.GetService<BotCoreTest::BotCore.Services.ITopstepXHttpClient>();
            var httpClient2 = serviceProvider.GetService<BotCoreTest::BotCore.Services.ITopstepXHttpClient>();
            
            Assert.Same(httpClient1, httpClient2);
        }

        [Fact]
        public async Task AuthService_WithCustomProvider_ShouldWork()
        {
            // Arrange
            var services = new ServiceCollection();
            services.AddLogging();
            
            var configuration = new ConfigurationBuilder().Build();
            var testJwt = "test.jwt.token";
            
            var customAuthProvider = (CancellationToken ct) => Task.FromResult(testJwt);

            services.AddTopstepAuthentication(configuration, customAuthProvider);
            var serviceProvider = services.BuildServiceProvider();

            // Act
            var authService = serviceProvider.GetRequiredService<BotCoreTest::BotCore.Auth.ITopstepAuth>();
            var (jwt, _) = await authService.GetFreshJwtAsync();

            // Assert
            Assert.Equal(testJwt, jwt);
        }
    }
}