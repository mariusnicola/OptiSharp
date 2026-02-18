using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class TruncatedNormalTests
{
    [Fact]
    public void LogPdf_AtMean_ReturnsMaxDensity()
    {
        var mu = 5.0;
        var sigma = 1.0;
        var logPdfAtMean = TruncatedNormal.LogPdf(mu, mu, sigma, 0, 10);
        var logPdfAway = TruncatedNormal.LogPdf(mu + 2, mu, sigma, 0, 10);
        Assert.True(logPdfAtMean > logPdfAway);
    }

    [Fact]
    public void LogPdf_OutsideBounds_ReturnsNegInfinity()
    {
        Assert.Equal(double.NegativeInfinity, TruncatedNormal.LogPdf(-1, 5, 1, 0, 10));
        Assert.Equal(double.NegativeInfinity, TruncatedNormal.LogPdf(11, 5, 1, 0, 10));
    }

    [Fact]
    public void LogPdf_WithinBounds_ReturnsFinite()
    {
        var result = TruncatedNormal.LogPdf(5.0, 5.0, 1.0, 0, 10);
        Assert.True(double.IsFinite(result));
        Assert.True(result > double.NegativeInfinity);
    }

    [Fact]
    public void Sample_AllWithinBounds_1000Samples()
    {
        var rng = new Random(42);
        for (var i = 0; i < 1000; i++)
        {
            var sample = TruncatedNormal.Sample(rng, 5.0, 1.0, 2.0, 8.0);
            Assert.InRange(sample, 2.0, 8.0);
        }
    }

    [Fact]
    public void Sample_MeanNearMu_WhenBoundsWide()
    {
        var rng = new Random(42);
        var sum = 0.0;
        var n = 10_000;
        for (var i = 0; i < n; i++)
            sum += TruncatedNormal.Sample(rng, 5.0, 1.0, 0, 10);

        var mean = sum / n;
        Assert.InRange(mean, 4.5, 5.5); // Should be near 5.0
    }

    [Fact]
    public void Sample_Deterministic_WithSeed()
    {
        var rng1 = new Random(123);
        var rng2 = new Random(123);

        for (var i = 0; i < 100; i++)
        {
            var s1 = TruncatedNormal.Sample(rng1, 5.0, 1.0, 0, 10);
            var s2 = TruncatedNormal.Sample(rng2, 5.0, 1.0, 0, 10);
            Assert.Equal(s1, s2);
        }
    }

    [Fact]
    public void LogSumExp_SingleValue_ReturnsSame()
    {
        ReadOnlySpan<double> values = [3.5];
        Assert.Equal(3.5, TruncatedNormal.LogSumExp(values), precision: 10);
    }

    [Fact]
    public void LogSumExp_TwoValues_Correct()
    {
        ReadOnlySpan<double> values = [1.0, 2.0];
        var expected = Math.Log(Math.Exp(1.0) + Math.Exp(2.0));
        Assert.Equal(expected, TruncatedNormal.LogSumExp(values), precision: 10);
    }

    [Fact]
    public void LogSumExp_LargeValues_NoOverflow()
    {
        ReadOnlySpan<double> values = [1000, 1001, 999];
        var result = TruncatedNormal.LogSumExp(values);
        Assert.True(double.IsFinite(result));
        Assert.True(result > 1000);
    }

    [Fact]
    public void LogSumExp_Empty_ReturnsNegInfinity()
    {
        Assert.Equal(double.NegativeInfinity, TruncatedNormal.LogSumExp(ReadOnlySpan<double>.Empty));
    }
}
