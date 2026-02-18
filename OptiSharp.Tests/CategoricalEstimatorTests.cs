using OptiSharp.Samplers.Tpe;

namespace OptiSharp.Tests;

public sealed class CategoricalEstimatorTests
{
    [Fact]
    public void Fit_UniformObservations_NearUniformWeights()
    {
        ReadOnlySpan<int> observations = [0, 1, 2, 0, 1, 2];
        var est = new CategoricalEstimator(observations, 3, priorWeight: 0.01);

        // All categories equally observed — log pdfs should be similar
        var diff01 = Math.Abs(est.LogPdf(0) - est.LogPdf(1));
        var diff12 = Math.Abs(est.LogPdf(1) - est.LogPdf(2));
        Assert.True(diff01 < 0.1);
        Assert.True(diff12 < 0.1);
    }

    [Fact]
    public void Fit_SingleCategory_Dominates()
    {
        ReadOnlySpan<int> observations = [0, 0, 0, 0, 0];
        var est = new CategoricalEstimator(observations, 3, priorWeight: 1.0);

        Assert.True(est.LogPdf(0) > est.LogPdf(1));
        Assert.True(est.LogPdf(0) > est.LogPdf(2));
    }

    [Fact]
    public void Fit_PriorPreventsZeroProbability()
    {
        ReadOnlySpan<int> observations = [0, 0, 0];
        var est = new CategoricalEstimator(observations, 3, priorWeight: 1.0);

        // Unobserved categories should still have non-zero probability
        Assert.True(est.LogPdf(1) > double.NegativeInfinity);
        Assert.True(est.LogPdf(2) > double.NegativeInfinity);
    }

    [Fact]
    public void Sample_RespectsWeights_1000Samples()
    {
        ReadOnlySpan<int> observations = [0, 0, 0, 0, 1];
        var est = new CategoricalEstimator(observations, 3, priorWeight: 0.01);
        var rng = new Random(42);

        var counts = new int[3];
        for (var i = 0; i < 1000; i++)
            counts[est.Sample(rng)]++;

        // Category 0 was observed 4x more than 1 — should be sampled much more
        Assert.True(counts[0] > counts[1] * 2);
        Assert.True(counts[0] > counts[2]);
    }

    [Fact]
    public void LogPdf_ProbabilitiesSumToOne()
    {
        ReadOnlySpan<int> observations = [0, 1, 2, 1];
        var est = new CategoricalEstimator(observations, 3, priorWeight: 1.0);

        var sum = Math.Exp(est.LogPdf(0)) + Math.Exp(est.LogPdf(1)) + Math.Exp(est.LogPdf(2));
        Assert.Equal(1.0, sum, precision: 6);
    }
}
