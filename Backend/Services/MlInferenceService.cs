using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Backend.DTOs;

namespace Backend.Services;

public interface IMlInferenceService
{
    Task<AnalyzeResponseDto> PredictAsync(string query);
    Task<string> GetMetricsAsync();
}

public class MlInferenceService : IMlInferenceService
{
    private readonly HttpClient _httpClient;

    public MlInferenceService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<AnalyzeResponseDto> PredictAsync(string query)
    {
        var response = await _httpClient.PostAsJsonAsync("/predict", new { query });
        response.EnsureSuccessStatusCode();

        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        };

        var mlResult = await response.Content.ReadFromJsonAsync<MlResultDto>(options);
        if (mlResult == null)
            throw new Exception("Received empty response from ML service");

        return new AnalyzeResponseDto
        {
            Query = mlResult.Query,
            Label = mlResult.Label,
            Confidence = mlResult.Confidence,
            IsSqlInjection = mlResult.IsSqlInjection || mlResult.Label == "SQL Injection",
            ScannedAt = DateTime.UtcNow
        };
    }

    public async Task<string> GetMetricsAsync()
    {
        var response = await _httpClient.GetAsync("/metrics");
        if (!response.IsSuccessStatusCode)
            throw new Exception($"ML service returned status: {response.StatusCode}");

        return await response.Content.ReadAsStringAsync();
    }

    private class MlResultDto
    {
        public string Query { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
        public double Confidence { get; set; }

        [JsonPropertyName("is_sql_injection")]
        public bool IsSqlInjection { get; set; }
    }
}
