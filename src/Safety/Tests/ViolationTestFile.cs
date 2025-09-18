using System;
using System.Threading.Tasks;

namespace TradingBot.Tests.ProductionEnforcement
{
    /// <summary>
    /// Test file to verify production enforcement analyzer catches violations
    /// This file intentionally contains patterns that should be blocked
    /// </summary>
    public class ViolationTestFile
    {
        // These should all be caught by the analyzer:
        
        // Hardcoded business value
        public decimal GetConfidenceThreshold() => 0.7m; // PRE011: Numeric literal
        
        // Mock pattern in name
        public class MockTradingService { } // PRE006: Placeholder pattern
        
        // Fixed size array
        public byte[] GetBuffer() => new byte[1024]; // PRE007: Fixed-size array
        
        // Empty async placeholder
        public async Task ProcessDataAsync()
        {
            await Task.Yield(); // PRE008: Empty async placeholder
            throw new NotImplementedException(); // PRE008: Empty async placeholder
        }
        
        /// <summary>
        /// Development pattern validation
        /// Verifies analyzer catches inappropriate development patterns in production code
        /// </summary>
        
        // Weak random
        public int GetRandomNumber() => new Random().Next(); // PRE010: Weak random
        
        // Hardcoded assignment
        public void ConfigureStrategy()
        {
            var riskFactor = 2.5; // PRE011: Numeric literal in business logic
            var positionSize = 0.25; // PRE011: Numeric literal in business logic
        }
    }
}