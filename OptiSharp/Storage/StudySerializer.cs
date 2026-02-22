using OptiSharp.Models;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OptiSharp.Storage;

internal sealed class StudySnapshot
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("direction")]
    public StudyDirection Direction { get; set; }

    [JsonPropertyName("directions")]
    public StudyDirection[]? Directions { get; set; }

    [JsonPropertyName("trials")]
    public List<TrialSnapshot> Trials { get; set; } = [];
}

internal sealed class TrialSnapshot
{
    [JsonPropertyName("number")]
    public int Number { get; set; }

    [JsonPropertyName("state")]
    public TrialState State { get; set; }

    [JsonPropertyName("value")]
    public double? Value { get; set; }

    [JsonPropertyName("values")]
    public double[]? Values { get; set; }

    [JsonPropertyName("parameters")]
    public Dictionary<string, JsonElement> Parameters { get; set; } = [];

    [JsonPropertyName("intermediateValues")]
    public Dictionary<int, double> IntermediateValues { get; set; } = [];

    [JsonPropertyName("constraintValues")]
    public double[]? ConstraintValues { get; set; }
}

/// <summary>
/// Internal serializer for persisting and loading Studies to/from JSON.
/// </summary>
internal static class StudySerializer
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = false,
        Converters = { new JsonStringEnumConverter() }
    };

    /// <summary>
    /// Serialize a study to a JSON snapshot (internal trials only).
    /// </summary>
    internal static string Serialize(
        string name,
        StudyDirection direction,
        StudyDirection[]? directions,
        IReadOnlyList<Trial> trials)
    {
        var snapshot = new StudySnapshot
        {
            Name = name,
            Direction = direction,
            Directions = directions,
            Trials = trials
                .Where(t => t.State == TrialState.Complete || t.State == TrialState.Pruned)
                .Select(t => new TrialSnapshot
                {
                    Number = t.Number,
                    State = t.State,
                    Value = t.Value,
                    Values = t.Values,
                    Parameters = t.Parameters.ToDictionary(
                        p => p.Key,
                        p => JsonSerializer.SerializeToElement(p.Value)
                    ),
                    IntermediateValues = t.IntermediateValues.ToDictionary(x => x.Key, x => x.Value),
                    ConstraintValues = t.ConstraintValues
                })
                .ToList()
        };

        return JsonSerializer.Serialize(snapshot, Options);
    }

    /// <summary>
    /// Deserialize a study from JSON, reconstructing parameter types using SearchSpace.
    /// </summary>
    internal static (string Name, StudyDirection[]? Directions, List<Trial> Trials) Deserialize(
        string json,
        SearchSpace searchSpace)
    {
        var snapshot = JsonSerializer.Deserialize<StudySnapshot>(json, Options)
            ?? throw new InvalidOperationException("Failed to deserialize study snapshot");

        var trials = new List<Trial>();

        foreach (var trialSnap in snapshot.Trials)
        {
            var parameters = ReconstructParameters(trialSnap.Parameters, searchSpace);
            var trial = new Trial(trialSnap.Number, parameters)
            {
                State = trialSnap.State,
                Value = trialSnap.Value,
                Values = trialSnap.Values,
                ConstraintValues = trialSnap.ConstraintValues
            };

            // Restore intermediate values
            foreach (var (step, value) in trialSnap.IntermediateValues)
                trial.Report(value, step);

            trials.Add(trial);
        }

        return (snapshot.Name, snapshot.Directions, trials);
    }

    private static Dictionary<string, object> ReconstructParameters(
        Dictionary<string, JsonElement> jsonParams,
        SearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>();

        // Build a lookup of ranges by name
        var rangesByName = searchSpace.ToDictionary(r => r.Name);

        foreach (var (paramName, jsonElement) in jsonParams)
        {
            if (!rangesByName.TryGetValue(paramName, out var range))
                throw new InvalidOperationException($"Parameter '{paramName}' not found in SearchSpace");

            object value = range switch
            {
                FloatRange _ => jsonElement.GetDouble(),
                IntRange _ => jsonElement.GetInt32(),
                CategoricalRange _ => jsonElement.GetString() ?? throw new InvalidOperationException($"Null categorical value for {paramName}"),
                _ => throw new InvalidOperationException($"Unknown range type for {paramName}")
            };

            parameters[paramName] = value;
        }

        return parameters;
    }
}
