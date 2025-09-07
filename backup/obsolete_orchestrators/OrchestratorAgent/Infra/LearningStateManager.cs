using System.Text.Json;

namespace OrchestratorAgent.Infra;

public static class LearningStateManager
{
    private static readonly string StateFile = Path.Combine("state", "learning_state.json");

    public static LearningState LoadState()
    {
        try
        {
            if (!File.Exists(StateFile))
            {
                return new LearningState();
            }

            var json = File.ReadAllText(StateFile);
            var state = JsonSerializer.Deserialize<LearningState>(json);
            return state ?? new LearningState();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LEARNING] Failed to load state: {ex.Message}");
            return new LearningState();
        }
    }

    public static void SaveState(LearningState state)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(StateFile)!);
            var json = JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(StateFile, json);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LEARNING] Failed to save state: {ex.Message}");
        }
    }
}

public class LearningState
{
    public DateTime LastPracticeUtc { get; set; } = DateTime.MinValue;
    public int CycleCount { get; set; } = 0;
    public string LastVersion { get; set; } = "1.0";
    public DateTime StateUpdatedUtc { get; set; } = DateTime.UtcNow;

    // Track learning metrics
    public double LastAccuracy { get; set; } = 0.0;
    public string LastRegime { get; set; } = "Unknown";
    public int ConsecutiveSuccesses { get; set; } = 0;
    public int TotalLearningCycles { get; set; } = 0;

    public TimeSpan TimeSinceLastPractice()
    {
        return DateTime.UtcNow - LastPracticeUtc;
    }

    public bool ShouldRunCycle(TimeSpan minGap)
    {
        return TimeSinceLastPractice() >= minGap;
    }

    public void RecordCycleCompletion()
    {
        LastPracticeUtc = DateTime.UtcNow;
        CycleCount++;
        TotalLearningCycles++;
        StateUpdatedUtc = DateTime.UtcNow;
    }
}
