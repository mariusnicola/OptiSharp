using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class ParzenEstimatorTests
{
    [Fact]
    public void Sample_AllWithinBounds()
    {
        var obs = new double[] { 2.0, 5.0, 8.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 1.0, considerMagicClip: true);
        var rng = new Random(42);

        var samples = est.Sample(rng, 1000);
        foreach (var s in samples)
            Assert.InRange(s, 0, 10);
    }

    [Fact]
    public void Sample_ConcentratesNearObservations()
    {
        var obs = new double[] { 5.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 0.1, considerMagicClip: true);
        var rng = new Random(42);

        var samples = est.Sample(rng, 1000);
        var nearObs = samples.Count(s => Math.Abs(s - 5.0) < 3.0);

        // Most samples should be near the single observation
        Assert.True(nearObs > 500, $"Expected >500 near obs, got {nearObs}");
    }

    [Fact]
    public void LogPdf_HigherNearObservations()
    {
        var obs = new double[] { 3.0, 7.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 0.1, considerMagicClip: true);

        var pdfNear = est.LogPdf([3.0]);
        var pdfFar = est.LogPdf([0.1]);

        Assert.True(pdfNear[0] > pdfFar[0]);
    }

    [Fact]
    public void LogPdf_UniformPrior_NonZeroEverywhere()
    {
        var obs = new double[] { 2.0, 8.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 1.0, considerMagicClip: true);

        // Test points far from observations — prior should keep density > 0
        var pdfs = est.LogPdf([0.1, 5.0, 9.9]);
        foreach (var pdf in pdfs)
            Assert.True(pdf > double.NegativeInfinity);
    }

    [Fact]
    public void SingleObservation_BandwidthSpansRange()
    {
        var obs = new double[] { 5.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 1.0, considerMagicClip: true);
        var rng = new Random(42);

        // With single obs and full-range bandwidth, should get spread samples
        var samples = est.Sample(rng, 500);
        var min = samples.Min();
        var max = samples.Max();

        Assert.True(max - min > 3.0, $"Expected spread > 3, got {max - min:F2}");
    }

    [Fact]
    public void TwoObservations_UseSpacing()
    {
        // With 2 widely-spaced observations and large range, bandwidths are wide.
        // Test that PDF is finite at all points and that observations contribute.
        var obs = new double[] { 2.0, 8.0 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 0.1, considerMagicClip: true);

        var pdf2 = est.LogPdf([2.0])[0];
        var pdf5 = est.LogPdf([5.0])[0];
        var pdf8 = est.LogPdf([8.0])[0];

        // All should be finite (no NaN or -Inf)
        Assert.True(double.IsFinite(pdf2));
        Assert.True(double.IsFinite(pdf5));
        Assert.True(double.IsFinite(pdf8));

        // Symmetric case: pdf at obs 2 ≈ pdf at obs 8 (by symmetry around midpoint)
        Assert.True(Math.Abs(pdf2 - pdf8) < 0.1, $"Expected symmetric: pdf2={pdf2:F4}, pdf8={pdf8:F4}");
    }

    [Fact]
    public void MagicClip_EnforcesMinimumBandwidth()
    {
        // Very close observations — without magic clip, bandwidth would be tiny
        var obs = new double[] { 5.0, 5.001, 5.002 };
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 0.1, considerMagicClip: true);
        var rng = new Random(42);

        // Should still get reasonable spread (not all samples at 5.0)
        var samples = est.Sample(rng, 500);
        var range = samples.Max() - samples.Min();
        Assert.True(range > 0.5, $"Expected range > 0.5 with magic clip, got {range:F4}");
    }

    [Fact]
    public void EmptyObservations_FallsBackToUniform()
    {
        var obs = Array.Empty<double>();
        var est = new ParzenEstimator(obs, 0, 10, priorWeight: 1.0, considerMagicClip: true);
        var rng = new Random(42);

        var samples = est.Sample(rng, 500);
        foreach (var s in samples)
            Assert.InRange(s, 0, 10);

        // Should be roughly uniform
        var mean = samples.Average();
        Assert.InRange(mean, 3.0, 7.0);
    }
}
