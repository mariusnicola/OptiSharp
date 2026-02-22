namespace OptiSharp.Pbt;

/// <summary>
/// A single member of the population in population-based training.
/// </summary>
public sealed record PbtMember(
    int Id,
    IReadOnlyDictionary<string, object> Parameters,
    double Performance,
    int Step,
    IReadOnlyList<IReadOnlyDictionary<string, object>> ParameterHistory);
