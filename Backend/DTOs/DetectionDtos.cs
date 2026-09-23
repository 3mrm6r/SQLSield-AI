namespace Backend.DTOs;

public class AnalyzeRequestDto
{
    public string Query { get; set; } = string.Empty;
}

public class AnalyzeResponseDto
{
    public int Id { get; set; }
    public string Query { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public double Confidence { get; set; }  
    public bool IsSqlInjection { get; set; }
    public DateTime ScannedAt { get; set; } = DateTime.UtcNow;
}
