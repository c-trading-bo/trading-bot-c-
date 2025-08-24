using System;
using System.Threading.Tasks;
using Xunit;

namespace OrchestratorAgent.Tests
{
    public class OrchestratorLaunchTests
    {
        [Fact]
        public async Task Program_StartsAndExitsQuickly_WithNoCredentials()
        {
            // Arrange: ensure quick-exit and no credentials to avoid network calls
            Environment.SetEnvironmentVariable("BOT_QUICK_EXIT", "1");
            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", null);
            Environment.SetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID", null);
            Environment.SetEnvironmentVariable("TOPSTEPX_USERNAME", null);
            Environment.SetEnvironmentVariable("TOPSTEPX_API_KEY", null);

            // Act & Assert: main should complete within a short time without throwing
            var start = DateTime.UtcNow;
            await OrchestratorAgent.Program.Main(Array.Empty<string>());
            var elapsed = DateTime.UtcNow - start;

            Assert.InRange(elapsed.TotalSeconds, 0, 15); // sanity guard to ensure it didn't hang
        }
    }
}