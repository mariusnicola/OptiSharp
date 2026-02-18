using System.Collections;

namespace OptiSharp.Models;

/// <summary>
/// Base type for parameter range definitions.
/// </summary>
public abstract record ParameterRange(string Name);

/// <summary>
/// Integer parameter range with optional step size.
/// </summary>
public sealed record IntRange(string Name, int Low, int High, int Step = 1) : ParameterRange(Name);

/// <summary>
/// Float parameter range with optional log-uniform sampling.
/// </summary>
public sealed record FloatRange(string Name, double Low, double High, bool Log = false) : ParameterRange(Name);

/// <summary>
/// Categorical parameter range with discrete choices.
/// </summary>
public sealed record CategoricalRange(string Name, object[] Choices) : ParameterRange(Name);

/// <summary>
/// Ordered collection of parameter ranges defining the search space.
/// </summary>
public sealed class SearchSpace : IReadOnlyList<ParameterRange>
{
    private readonly List<ParameterRange> _ranges;
    private readonly Dictionary<string, int> _nameIndex;

    public SearchSpace(IEnumerable<ParameterRange> ranges)
    {
        _ranges = ranges.ToList();
        _nameIndex = new Dictionary<string, int>(_ranges.Count);
        for (var i = 0; i < _ranges.Count; i++)
        {
            if (_nameIndex.ContainsKey(_ranges[i].Name))
                throw new ArgumentException($"Duplicate parameter name: '{_ranges[i].Name}'");
            _nameIndex[_ranges[i].Name] = i;
        }
    }

    public int Count => _ranges.Count;
    public ParameterRange this[int index] => _ranges[index];
    public ParameterRange this[string name] => _ranges[_nameIndex[name]];
    public bool Contains(string name) => _nameIndex.ContainsKey(name);

    public IEnumerator<ParameterRange> GetEnumerator() => _ranges.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
