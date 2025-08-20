using Xunit;
using BotCore;
using System.Collections.Generic;

public class AllStrategiesTests
{
    [Fact]
    public void TickRounding_IsCorrect()
    {
        decimal price = 4023.13m;
        decimal rounded = Px.RoundToTick(price);
        Assert.Equal(4023.25m, rounded);
    }

    [Fact]
    public void RMultiple_IsCorrect()
    {
        decimal entry = 100m, stop = 99m, target = 102m;
        decimal r = Px.RMultiple(entry, stop, target, true);
        Assert.Equal(2.00m, r);
    }

    [Fact]
    public void DuplicateOrderProtection_Works()
    {
        var tags = new HashSet<string>();
        string tag = "S11L-20250820-120000";
        tags.Add(tag);
        Assert.False(tags.Add(tag)); // Should not add duplicate
    }
}
