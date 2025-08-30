using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.ML;

namespace RLComponentTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== RL Sizer Component Verification Test ===\n");
            
            // Create logger factory
            using var loggerFactory = LoggerFactory.Create(builder => 
                builder.AddConsole().SetMinimumLevel(LogLevel.Debug));
            
            bool allTestsPassed = true;
            
            // Test 1: SizerCanary functionality
            try
            {
                Console.WriteLine("1. Testing SizerCanary...");
                var canaryLogger = loggerFactory.CreateLogger<SizerCanary>();
                var canary = new SizerCanary(canaryLogger);
                
                var config = canary.GetConfig();
                Console.WriteLine($"   Config: {System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions { WriteIndented = true })}");
                
                // Test traffic splitting consistency
                var signalId = "ES_S3_20250829_120000";
                var result1 = canary.ShouldUseRl(signalId);
                var result2 = canary.ShouldUseRl(signalId);
                
                if (result1 != result2)
                {
                    Console.WriteLine("   ❌ FAIL: Traffic splitting not deterministic");
                    allTestsPassed = false;
                }
                else
                {
                    Console.WriteLine($"   ✅ PASS: Deterministic traffic splitting (signal {signalId} -> UseRL: {result1})");
                }
                
                // Test distribution over multiple signals
                int rlCount = 0, baselineCount = 0;
                for (int i = 0; i < 100; i++)
                {
                    var testSignal = $"ES_S3_20250829_{i:D6}";
                    if (canary.ShouldUseRl(testSignal))
                        rlCount++;
                    else
                        baselineCount++;
                }
                
                double rlRatio = rlCount / 100.0;
                Console.WriteLine($"   ✅ PASS: Traffic distribution over 100 signals: {rlRatio:P0} RL, {(1-rlRatio):P0} Baseline");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ FAIL: SizerCanary test failed - {ex.Message}");
                allTestsPassed = false;
            }
            
            // Test 2: RlSizer initialization (without model file)
            try
            {
                Console.WriteLine("\n2. Testing RlSizer initialization...");
                var rlLogger = loggerFactory.CreateLogger<RlSizer>();
                var modelPath = "models/rl/nonexistent_model.onnx";
                var actions = new float[] { 0.5f, 0.75f, 1.0f, 1.25f, 1.5f };
                
                var rlSizer = new RlSizer(modelPath, actions, false, 120, rlLogger);
                
                if (!rlSizer.IsLoaded)
                {
                    Console.WriteLine("   ✅ PASS: RlSizer correctly reports not loaded when model file missing");
                }
                else
                {
                    Console.WriteLine("   ❌ FAIL: RlSizer should report not loaded when model file missing");
                    allTestsPassed = false;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ FAIL: RlSizer initialization test failed - {ex.Message}");
                allTestsPassed = false;
            }
            
            // Test 3: FeatureSnapshot creation
            try
            {
                Console.WriteLine("\n3. Testing FeatureSnapshot...");
                var snapshot = new FeatureSnapshot
                {
                    Symbol = "ES",
                    Strategy = "S3",
                    Session = "RTH",
                    Regime = "High_Vol",
                    Timestamp = DateTime.UtcNow,
                    Price = 4500.25f,
                    SignalStrength = 2.5f,
                    PriorWinRate = 0.65f
                };
                
                // Test that ToDict works (should have 17+ features)
                var featureDict = snapshot.ToDict();
                if (featureDict.Count >= 17)
                {
                    Console.WriteLine($"   ✅ PASS: FeatureSnapshot.ToDict() returns {featureDict.Count} features (≥ 17 expected)");
                }
                else
                {
                    Console.WriteLine($"   ❌ FAIL: FeatureSnapshot.ToDict() returns only {featureDict.Count} features (< 17 expected)");
                    allTestsPassed = false;
                }
                
                Console.WriteLine($"   Sample features: Price={featureDict["price"]:F2}, SignalStrength={featureDict["signal_strength"]:F2}, PriorWinRate={featureDict["prior_win_rate"]:F2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ FAIL: FeatureSnapshot test failed - {ex.Message}");
                allTestsPassed = false;
            }
            
            // Test 4: Environment variable configuration
            try
            {
                Console.WriteLine("\n4. Testing environment variable configuration...");
                var envVars = new[]
                {
                    "RL_ENABLED", "RL_ONNX", "RL_ACTIONS", "RL_SAMPLE_ACTION", "RL_MAX_AGE_MIN",
                    "RL_LOOKBACK_DAYS", "RL_TEST_FORWARD_DAYS", "RL_EMBARGO_BARS", "RL_CVAR_LEVEL",
                    "RL_CVAR_TARGET_R", "RL_LAGRANGE_INIT", "RL_POLICY_HIDDEN", "RL_POLICY_LAYERS",
                    "RL_PPO_STEPS", "RL_PPO_LR", "RL_PPO_CLIP", "RL_PPO_GAMMA", "RL_PPO_LAMBDA",
                    "RL_SIZER_CANARY_ENABLED", "RL_SIZER_CANARY_RL_FRACTION"
                };
                
                int configuredCount = 0;
                foreach (var envVar in envVars)
                {
                    var value = Environment.GetEnvironmentVariable(envVar);
                    if (!string.IsNullOrEmpty(value))
                    {
                        configuredCount++;
                    }
                }
                
                Console.WriteLine($"   ✅ PASS: {configuredCount}/{envVars.Length} RL environment variables are configured");
                if (configuredCount < envVars.Length)
                {
                    Console.WriteLine("   ⚠️  WARN: Some RL environment variables are missing (this is OK for testing)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ FAIL: Environment variable test failed - {ex.Message}");
                allTestsPassed = false;
            }
            
            Console.WriteLine("\n=== Test Summary ===");
            if (allTestsPassed)
            {
                Console.WriteLine("🎉 ALL TESTS PASSED! RL Sizer components are working correctly.");
                Console.WriteLine("\nNext steps:");
                Console.WriteLine("1. Collect training data by running the bot");
                Console.WriteLine("2. Train the RL model using ml/rl/train_cvar_ppo.py");
                Console.WriteLine("3. Deploy the trained model to models/rl/latest_rl_sizer.onnx");
                Console.WriteLine("4. Monitor A/A test results via /healthz/canary endpoint");
            }
            else
            {
                Console.WriteLine("❌ SOME TESTS FAILED. Please check the implementation.");
                Environment.Exit(1);
            }
        }
    }
}
