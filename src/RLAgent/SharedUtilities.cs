namespace TradingBot.RLAgent;

/// <summary>
/// Shared utilities and common classes for RL Agent components
/// </summary>

/// <summary>
/// Circular buffer for efficient data storage
/// </summary>
public class CircularBuffer<T>
{
    private readonly T[] _buffer;
    private readonly int _size;
    private int _head;
    private int _count;

    public CircularBuffer(int size)
    {
        _size = size;
        _buffer = new T[size];
    }

    public int Count => _count;

    public void Add(T item)
    {
        _buffer[_head] = item;
        _head = (_head + 1) % _size;
        if (_count < _size)
            _count++;
    }

    public T? GetFromEnd(int index)
    {
        if (index >= _count) return default(T);
        var actualIndex = (_head - 1 - index + _size) % _size;
        return _buffer[actualIndex];
    }

    public T[] GetLast(int count)
    {
        count = Math.Min(count, _count);
        var result = new T[count];
        
        for (int i = 0; i < count; i++)
        {
            var item = GetFromEnd(i);
            if (!EqualityComparer<T>.Default.Equals(item, default(T)))
                result[count - 1 - i] = item!;
        }
        
        return result;
    }

    public T[] GetAll()
    {
        var result = new T[_count];
        for (int i = 0; i < _count; i++)
        {
            var index = (_head - _count + i + _size) % _size;
            result[i] = _buffer[index];
        }
        return result;
    }
}

/// <summary>
/// Market data structure
/// </summary>
public class MarketData
{
    public DateTime Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
}

/// <summary>
/// Regime types for ML/RL components
/// </summary>
public enum RegimeType
{
    Range = 0,
    Trend = 1,
    Volatility = 2,
    LowVol = 3,
    HighVol = 4
}