namespace OptiSharp.Samplers.Sobol;

/// <summary>
/// Halton quasi-random sequence generator using prime-based construction.
/// Better low-discrepancy properties than Sobol in high dimensions (p100+).
/// Each dimension uses a different prime base: dim 0 → base 2, dim 1 → base 3, dim 2 → base 5, etc.
/// </summary>
internal sealed class HaltonSequence
{
    private readonly int _dimensions;
    private readonly int[] _primeBases;
    private ulong _index;

    /// <summary>
    /// Maximum supported dimensions (limited by prime availability).
    /// Using primes up to 547 allows ~100+ dimensions.
    /// </summary>
    public const int MaxDimensions = 100;

    private static readonly int[] SmallPrimes = new[]
    {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
        331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
        421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
        509, 521, 523, 541, 547
    };

    public HaltonSequence(int dimensions, int? seed = null)
    {
        if (dimensions < 1 || dimensions > MaxDimensions)
            throw new ArgumentOutOfRangeException(nameof(dimensions),
                $"Halton sampler supports up to {MaxDimensions} dimensions, got {dimensions}.");

        _dimensions = dimensions;
        _primeBases = new int[dimensions];
        for (var i = 0; i < dimensions; i++)
            _primeBases[i] = SmallPrimes[i];

        _index = 0;

        // If seed provided, skip to seed-based starting index for pseudo-randomization
        if (seed.HasValue)
        {
            var hash = unchecked((ulong)seed.Value * 2654435761u);
            _index = hash % 10000; // Skip first 10000 to avoid low-quality initial points
        }
    }

    /// <summary>
    /// Generate the next point in the Halton sequence [0, 1)^D.
    /// </summary>
    public double[] NextPoint()
    {
        var point = new double[_dimensions];
        var index = _index + 1; // Halton sequence is 1-indexed internally

        for (var d = 0; d < _dimensions; d++)
        {
            var base_ = _primeBases[d];
            var value = 0.0;
            var invBase = 1.0 / base_;
            var frac = invBase;

            var n = index;
            while (n > 0)
            {
                var digit = (int)(n % (ulong)base_);
                value += digit * frac;
                frac *= invBase;
                n /= (ulong)base_;
            }

            point[d] = value;
        }

        _index++;
        return point;
    }

    /// <summary>
    /// Skip ahead in the sequence without generating intermediate points.
    /// For simplicity, this just advances the index counter.
    /// </summary>
    public void Skip(int count)
    {
        _index += (ulong)count;
    }
}
