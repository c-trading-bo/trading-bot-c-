using System;
using System.Collections.Generic;
using System.Linq;

namespace Trading.Strategies;

/// <summary>
/// Analytics utilities for trading strategies, focusing on returns-based calculations
/// </summary>
public static class Analytics
{
    /// <summary>
    /// Calculates Pearson correlation coefficient between two series of rolling returns
    /// </summary>
    /// <param name="returnsX">First returns series</param>
    /// <param name="returnsY">Second returns series</param>
    /// <returns>Pearson correlation coefficient (-1 to 1), or NaN if calculation is not possible</returns>
    public static double CalculatePearsonCorrelation(IEnumerable<double> returnsX, IEnumerable<double> returnsY)
    {
        ArgumentNullException.ThrowIfNull(returnsX);
        ArgumentNullException.ThrowIfNull(returnsY);
            
        var arrayX = returnsX.ToArray();
        var arrayY = returnsY.ToArray();
        
        if (arrayX.Length != arrayY.Length)
            throw new ArgumentException("Returns series must have the same length");
            
        const int minSampleSize = 2; // Minimum samples needed for correlation calculation
        if (arrayX.Length < minSampleSize)
            return double.NaN;
            
        // Calculate means
        var meanX = arrayX.Average();
        var meanY = arrayY.Average();
        
        // Calculate numerator and denominators
        const double initialValue = 0.0; // Initial accumulator values
        var numerator = initialValue;
        var sumSquareX = initialValue;
        var sumSquareY = initialValue;
        
        for (int i = 0; i < arrayX.Length; i++)
        {
            var deviationX = arrayX[i] - meanX;
            var deviationY = arrayY[i] - meanY;
            
            numerator += deviationX * deviationY;
            sumSquareX += deviationX * deviationX;
            sumSquareY += deviationY * deviationY;
        }
        
        var denominator = Math.Sqrt(sumSquareX * sumSquareY);
        
        if (Math.Abs(denominator) < double.Epsilon)
            return double.NaN;
            
        return numerator / denominator;
    }
    
    /// <summary>
    /// Calculates rolling returns from a price series
    /// </summary>
    /// <param name="prices">Price series</param>
    /// <param name="periods">Number of periods for return calculation (default 1 for period-over-period)</param>
    /// <returns>Rolling returns series</returns>
    public static IEnumerable<double> CalculateRollingReturns(IEnumerable<double> prices, int periods = 1)
    {
        ArgumentNullException.ThrowIfNull(prices);
            
        if (periods <= 0)
            throw new ArgumentException("Periods must be positive", nameof(periods));
            
        return CalculateRollingReturnsIterator(prices, periods);
    }
    
    private static IEnumerable<double> CalculateRollingReturnsIterator(IEnumerable<double> prices, int periods)
    {
        var priceArray = prices.ToArray();
        
        if (priceArray.Length <= periods)
            yield break;
            
        for (int i = periods; i < priceArray.Length; i++)
        {
            var currentPrice = priceArray[i];
            var previousPrice = priceArray[i - periods];
            
            if (Math.Abs(previousPrice) < double.Epsilon)
            {
                const double zeroDivisionFallback = 0.0; // Return for zero division cases
                yield return zeroDivisionFallback;
            }
            else
            {
                yield return (currentPrice - previousPrice) / previousPrice;
            }
        }
    }
    
    /// <summary>
    /// Calculates rolling correlation between two price series using their returns
    /// </summary>
    /// <param name="pricesX">First price series</param>
    /// <param name="pricesY">Second price series</param>
    /// <param name="windowSize">Rolling window size for correlation calculation</param>
    /// <param name="returnPeriods">Periods for return calculation</param>
    /// <returns>Rolling correlation series</returns>
    public static IEnumerable<double> CalculateRollingCorrelation(
        IEnumerable<double> pricesX, 
        IEnumerable<double> pricesY, 
        int windowSize, 
        int returnPeriods = 1)
    {
        var returnsX = CalculateRollingReturns(pricesX, returnPeriods).ToArray();
        var returnsY = CalculateRollingReturns(pricesY, returnPeriods).ToArray();
        
        if (returnsX.Length != returnsY.Length || returnsX.Length < windowSize)
            yield break;
            
        for (int i = windowSize - 1; i < returnsX.Length; i++)
        {
            var windowReturnsX = returnsX.Skip(i - windowSize + 1).Take(windowSize);
            var windowReturnsY = returnsY.Skip(i - windowSize + 1).Take(windowSize);
            
            yield return CalculatePearsonCorrelation(windowReturnsX, windowReturnsY);
        }
    }
    
    /// <summary>
    /// Calculates the Sharpe ratio from a returns series
    /// </summary>
    /// <param name="returns">Returns series</param>
    /// <param name="riskFreeRate">Risk-free rate (annualized)</param>
    /// <param name="periodsPerYear">Number of periods per year for annualization</param>
    /// <returns>Sharpe ratio</returns>
    public static double CalculateSharpeRatio(IEnumerable<double> returns, double riskFreeRate = 0.0, double periodsPerYear = 252.0)
    {
        var returnsArray = returns.ToArray();
        
        if (returnsArray.Length < 2)
            return double.NaN;
            
        var meanReturn = returnsArray.Average();
        var stdDev = Math.Sqrt(returnsArray.Select(r => Math.Pow(r - meanReturn, 2)).Average());
        
        if (Math.Abs(stdDev) < double.Epsilon)
            return double.NaN;
            
        var annualizedReturn = meanReturn * periodsPerYear;
        var annualizedStdDev = stdDev * Math.Sqrt(periodsPerYear);
        
        return (annualizedReturn - riskFreeRate) / annualizedStdDev;
    }
}