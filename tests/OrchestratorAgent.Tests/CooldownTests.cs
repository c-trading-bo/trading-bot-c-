using Xunit;

namespace OrchestratorAgent.Tests
{
    public class CooldownTests
    {
        [Fact]
        public void CooldownPreventsIdenticalRefire()
        {
            var ok1 = BotCore.RecentSignalCache.ShouldEmit("S1", "ES", "BUY", 5000m, 5010m, 4990m, 30);
            var ok2 = BotCore.RecentSignalCache.ShouldEmit("S1", "ES", "BUY", 5000m, 5010m, 4990m, 30);
            Assert.True(ok1);
            Assert.False(ok2);
        }
    }
}
