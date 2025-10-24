using System;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Reflection;
using Xunit;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// Tests for Model Registry Bootstrap Service
    /// Validates that S15_RL Policy, HistoricalPatternRecognitionService, and PositionManagementOptimizer
    /// are properly registered in the champion/challenger tracking system
    /// </summary>
    public class ModelRegistryBootstrapTests
    {
        /// <summary>
        /// Test that ModelRegistryBootstrapService is properly registered as IHostedService
        /// and contains the algorithm names for the 3 critical components
        /// </summary>
        [Fact]
        public void BootstrapService_ContainsAllThreeCriticalAlgorithms()
        {
            // This test validates that the ModelRegistryBootstrapService source code
            // contains registrations for the 3 critical components
            
            // Read the source file
            var bootstrapFilePath = "../../../../../src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs";
            if (!System.IO.File.Exists(bootstrapFilePath))
            {
                // Try relative path from test bin directory
                bootstrapFilePath = "../../../../../../src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs";
            }
            
            Assert.True(System.IO.File.Exists(bootstrapFilePath), "ModelRegistryBootstrapService.cs file should exist");
            
            var sourceCode = System.IO.File.ReadAllText(bootstrapFilePath);
            
            // Verify all 3 critical components are in the source
            Assert.Contains("S15-RL-Policy", sourceCode);
            Assert.Contains("Pattern-Recognition", sourceCode);
            Assert.Contains("PM-Optimizer", sourceCode);
            
            // Verify they are in the algorithm registration array
            Assert.Contains("(\"S15-RL-Policy\",", sourceCode);
            Assert.Contains("(\"Pattern-Recognition\",", sourceCode);
            Assert.Contains("(\"PM-Optimizer\",", sourceCode);
        }
        
        /// <summary>
        /// Test that the bootstrap service includes all 9 expected algorithms
        /// </summary>
        [Fact]
        public void BootstrapService_ContainsAllNineAlgorithms()
        {
            var bootstrapFilePath = "../../../../../src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs";
            if (!System.IO.File.Exists(bootstrapFilePath))
            {
                bootstrapFilePath = "../../../../../../src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs";
            }
            
            Assert.True(System.IO.File.Exists(bootstrapFilePath), "ModelRegistryBootstrapService.cs file should exist");
            
            var sourceCode = System.IO.File.ReadAllText(bootstrapFilePath);
            
            // All 9 expected algorithms
            var expectedAlgorithms = new[]
            {
                "CVaR-PPO",
                "Neural-UCB",
                "Regime-Detector",
                "Model-Ensemble",
                "Online-Learning-System",
                "Slippage-Latency-Model",
                "S15-RL-Policy",
                "Pattern-Recognition",
                "PM-Optimizer"
            };
            
            foreach (var algorithm in expectedAlgorithms)
            {
                Assert.Contains($"\"{algorithm}\"", sourceCode);
            }
        }
        
        /// <summary>
        /// Test that Program.cs properly registers the ModelRegistryBootstrapService as IHostedService
        /// </summary>
        [Fact]
        public void ProgramCs_RegistersBootstrapServiceAsHostedService()
        {
            var programFilePath = "../../../../../src/UnifiedOrchestrator/Program.cs";
            if (!System.IO.File.Exists(programFilePath))
            {
                programFilePath = "../../../../../../src/UnifiedOrchestrator/Program.cs";
            }
            
            Assert.True(System.IO.File.Exists(programFilePath), "Program.cs file should exist");
            
            var sourceCode = System.IO.File.ReadAllText(programFilePath);
            
            // Verify bootstrap service is registered as IHostedService
            Assert.Contains("AddHostedService<ModelRegistryBootstrapService>()", sourceCode);
        }
        
        /// <summary>
        /// Test that the 3 critical services are also registered in the DI container
        /// </summary>
        [Fact]
        public void ProgramCs_RegistersThreeCriticalServicesInDI()
        {
            var programFilePath = "../../../../../src/UnifiedOrchestrator/Program.cs";
            if (!System.IO.File.Exists(programFilePath))
            {
                programFilePath = "../../../../../../src/UnifiedOrchestrator/Program.cs";
            }
            
            Assert.True(System.IO.File.Exists(programFilePath), "Program.cs file should exist");
            
            var sourceCode = System.IO.File.ReadAllText(programFilePath);
            
            // Verify the 3 services are registered in DI
            Assert.Contains("PositionManagementOptimizer", sourceCode);
            Assert.Contains("HistoricalPatternRecognitionService", sourceCode);
            Assert.Contains("OnnxRlPolicy", sourceCode);
        }
    }
}
