using System;

// Simple console app to run the 24/7 trading system integration test
public static class Program
{
    public static void Main(string[] args)
    {
        try
        {
            TestEnhancedZones.IntegrationTest.TestTimeOptimizedTrading();
            Console.WriteLine("\n🎉 Integration test completed successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Integration test failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        
        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
}