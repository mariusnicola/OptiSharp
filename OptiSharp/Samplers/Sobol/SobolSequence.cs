namespace OptiSharp.Samplers.Sobol;

/// <summary>
/// Sobol quasi-random sequence generator using Joe-Kuo direction numbers.
/// Supports up to 200 dimensions with optional XOR scrambling for seed-based randomization.
/// Uses Gray code construction for O(1) per-point generation.
/// </summary>
internal sealed class SobolSequence
{
    private const int Bits = 32;
    private const double Normalizer = 1.0 / (1L << Bits);

    private readonly int _dimensions;
    private readonly uint[][] _directionNumbers; // [dimension][bit]
    private readonly uint[] _x;                  // Current state per dimension
    private readonly uint[]? _scrambleMasks;      // XOR scramble mask per dimension
    private uint _index;                          // Gray code counter (0 = first point not yet generated)

    /// <summary>
    /// Maximum supported dimensions (limited by Joe-Kuo direction number table).
    /// </summary>
    public const int MaxDimensions = 200;

    public SobolSequence(int dimensions, int? seed = null)
    {
        if (dimensions < 1 || dimensions > MaxDimensions)
            throw new ArgumentOutOfRangeException(nameof(dimensions),
                $"Dimensions must be between 1 and {MaxDimensions}, got {dimensions}.");

        _dimensions = dimensions;
        _directionNumbers = new uint[dimensions][];
        _x = new uint[dimensions];

        InitializeDirectionNumbers();

        if (seed.HasValue)
            _scrambleMasks = GenerateScrambleMasks(dimensions, seed.Value);
    }

    /// <summary>
    /// Generate the next point in the Sobol sequence.
    /// Returns values in [0, 1) for each dimension.
    /// </summary>
    public double[] NextPoint()
    {
        if (_index == 0)
        {
            // First point is the origin (0, 0, ..., 0) — or scrambled version
            _index++;
            var point = new double[_dimensions];
            for (var d = 0; d < _dimensions; d++)
            {
                var val = _scrambleMasks != null ? _x[d] ^ _scrambleMasks[d] : _x[d];
                point[d] = val * Normalizer;
            }
            return point;
        }

        var c = RightmostZeroBit(_index - 1);
        _index++;

        var result = new double[_dimensions];
        for (var d = 0; d < _dimensions; d++)
        {
            _x[d] ^= _directionNumbers[d][c];
            var val = _scrambleMasks != null ? _x[d] ^ _scrambleMasks[d] : _x[d];
            result[d] = val * Normalizer;
        }
        return result;
    }

    /// <summary>
    /// Skip n points in the sequence.
    /// </summary>
    public void Skip(int n)
    {
        for (var i = 0; i < n; i++)
            NextPoint();
    }

    private void InitializeDirectionNumbers()
    {
        // Dimension 0: Van der Corput base-2 sequence
        _directionNumbers[0] = new uint[Bits];
        for (var i = 0; i < Bits; i++)
            _directionNumbers[0][i] = 1u << (Bits - 1 - i);

        // Dimensions 1+: Use Joe-Kuo direction numbers
        for (var d = 1; d < _dimensions; d++)
        {
            var v = new uint[Bits];
            var poly = JoeKuoData.Polynomials[d - 1];
            var s = HighestBitPosition(poly); // Degree of primitive polynomial

            // Initial direction numbers from Joe-Kuo table
            var initNums = JoeKuoData.InitialDirectionNumbers[d - 1];
            for (var i = 0; i < s; i++)
                v[i] = initNums[i] << (Bits - 1 - i);

            // Recurrence for remaining direction numbers
            // V[i] = c_1*V[i-1] XOR c_2*V[i-2] XOR ... XOR c_{s-1}*V[i-s+1] XOR V[i-s] XOR (V[i-s] >> s)
            for (var i = s; i < Bits; i++)
            {
                var vi = v[i - s] ^ (v[i - s] >> s);
                for (var k = 1; k < s; k++)
                {
                    if (((poly >> k) & 1) == 1)
                        vi ^= v[i - k];
                }
                v[i] = vi;
            }

            _directionNumbers[d] = v;
        }
    }

    /// <summary>
    /// Position of highest set bit (= degree of the primitive polynomial).
    /// poly includes leading 1 bit, so for poly=3 (0b11) → 1, poly=7 (0b111) → 2.
    /// </summary>
    private static int HighestBitPosition(uint poly)
    {
        var s = 0;
        while ((1u << (s + 1)) <= poly) s++;
        return s;
    }

    /// <summary>
    /// Position of rightmost zero bit (for Gray code construction).
    /// </summary>
    private static int RightmostZeroBit(uint n)
    {
        var pos = 0;
        while ((n & 1) == 1)
        {
            n >>= 1;
            pos++;
        }
        return pos;
    }

    /// <summary>
    /// Generate per-dimension XOR scramble masks from a seed.
    /// Uses Matousek-style digital shift scrambling.
    /// </summary>
    private static uint[] GenerateScrambleMasks(int dimensions, int seed)
    {
        var rng = new Random(seed);
        var masks = new uint[dimensions];
        for (var d = 0; d < dimensions; d++)
        {
            // Generate a random 32-bit mask
            masks[d] = (uint)(rng.Next() << 16) | (uint)(rng.Next() & 0xFFFF);
        }
        return masks;
    }
}
