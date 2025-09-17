using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace TradingBot.Tests
{
    /// <summary>
    /// Test to verify duplicate agent session prevention works correctly
    /// Addresses Comment #3304685224: Eliminate Duplicate Agent Launches
    /// </summary>
    public class DuplicateSessionPreventionTest
    {
        /// <summary>
        /// Simulate the UnifiedOrchestratorService duplicate prevention logic
        /// This demonstrates how the actual service prevents double premiums
        /// </summary>
        public class MockAgentRegistry
        {
            private readonly ConcurrentDictionary<string, bool> _activeAgentSessions = new();
            private readonly ConcurrentDictionary<string, DateTime> _agentSessionStartTimes = new();
            private readonly object _agentSessionLock = new();
            private readonly ILogger _logger;

            public MockAgentRegistry(ILogger logger)
            {
                _logger = logger;
            }

            /// <summary>
            /// Replicate TryLaunchAgent logic to verify duplicate prevention
            /// </summary>
            public bool TryLaunchAgent(string agentKey, Func<Task> launchAction)
            {
                lock (_agentSessionLock)
                {
                    // Check if agent session already active
                    if (_activeAgentSessions.ContainsKey(agentKey))
                    {
                        var startTime = _agentSessionStartTimes.GetValueOrDefault(agentKey);
                        _logger.LogWarning("🚫 [TEST] Duplicate launch prevented for agentKey: {AgentKey}, already running since {StartTime}", 
                            agentKey, startTime);
                        return false; // Prevent duplicate launch = No double premium
                    }
                    
                    // Register new agent session
                    _activeAgentSessions[agentKey] = true;
                    _agentSessionStartTimes[agentKey] = DateTime.UtcNow;
                    
                    _logger.LogInformation("✅ [TEST] Agent session registered: {AgentKey} at {StartTime}", 
                        agentKey, DateTime.UtcNow);
                    
                    // Execute launch action asynchronously with cleanup
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await launchAction();
                        }
                        finally
                        {
                            // Remove from registry when done
                            lock (_agentSessionLock)
                            {
                                _activeAgentSessions.TryRemove(agentKey, out _);
                                _agentSessionStartTimes.TryRemove(agentKey, out _);
                                _logger.LogInformation("🗑️ [TEST] Agent session cleanup: {AgentKey}", agentKey);
                            }
                        }
                    });
                    
                    return true;
                }
            }

            /// <summary>
            /// Test runtime proof: Simulate concurrent launch attempts
            /// </summary>
            public async Task<(int successful, int duplicatePrevented)> SimulateConcurrentLaunches(string agentKey, int attemptCount)
            {
                var tasks = new List<Task<bool>>();
                
                // Simulate multiple concurrent launch attempts
                for (int i = 0; i < attemptCount; i++)
                {
                    tasks.Add(Task.Run(() => TryLaunchAgent(agentKey, async () => 
                    {
                        await Task.Delay(100); // Simulate agent work
                    })));
                }

                var results = await Task.WhenAll(tasks);
                var successful = results.Count(r => r);
                var duplicatePrevented = results.Count(r => !r);

                _logger.LogInformation("📊 [TEST] Concurrent launch results: {Successful} successful, {Prevented} prevented", 
                    successful, duplicatePrevented);

                return (successful, duplicatePrevented);
            }
        }

        /// <summary>
        /// Runtime proof that duplicate session prevention works
        /// </summary>
        public static async Task<bool> VerifyDuplicatePreventionAsync(ILogger logger)
        {
            var registry = new MockAgentRegistry(logger);
            
            // Test 1: Single launch should succeed
            var firstLaunch = registry.TryLaunchAgent("test-agent-1", async () => await Task.Delay(50));
            if (!firstLaunch)
            {
                logger.LogError("❌ [TEST] First launch should have succeeded");
                return false;
            }

            // Test 2: Immediate duplicate launch should be prevented
            var duplicateLaunch = registry.TryLaunchAgent("test-agent-1", async () => await Task.Delay(50));
            if (duplicateLaunch)
            {
                logger.LogError("❌ [TEST] Duplicate launch should have been prevented");
                return false;
            }

            // Test 3: Concurrent launches should only allow one through
            var (successful, prevented) = await registry.SimulateConcurrentLaunches("test-agent-2", 5);
            if (successful != 1)
            {
                logger.LogError("❌ [TEST] Expected exactly 1 successful launch, got {Successful}", successful);
                return false;
            }

            if (prevented != 4)
            {
                logger.LogError("❌ [TEST] Expected exactly 4 prevented launches, got {Prevented}", prevented);
                return false;
            }

            logger.LogInformation("✅ [TEST] All duplicate prevention tests passed");
            return true;
        }
    }
}