using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Backend.Data;
using Backend.DTOs;
using Backend.Services;

namespace Backend.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DetectionController : ControllerBase
{
    private readonly IMlInferenceService _mlService;
    private readonly AppDbContext _db;

    public DetectionController(IMlInferenceService mlService, AppDbContext db)
    {
        _mlService = mlService;
        _db = db;
    }

    [HttpPost("analyze")]
    public async Task<ActionResult<AnalyzeResponseDto>> Analyze([FromBody] AnalyzeRequestDto request)
    {
        if (string.IsNullOrWhiteSpace(request.Query))
            return BadRequest("Query cannot be empty.");

        var result = await _mlService.PredictAsync(request.Query);

        // Save scan log to database
        var log = new ScanLog
        {
            Query = result.Query,
            Label = result.Label,
            Confidence = result.Confidence,
            IsSqlInjection = result.IsSqlInjection,
            ScannedAt = DateTime.UtcNow
        };
        _db.ScanLogs.Add(log);
        await _db.SaveChangesAsync();

        result.Id = log.Id;
        return Ok(result);
    }

    [HttpGet("history")]
    public async Task<ActionResult<IEnumerable<ScanLog>>> GetHistory()
    {
        var logs = await _db.ScanLogs.OrderByDescending(x => x.ScannedAt).Take(20).ToListAsync();
        return Ok(logs);
    }

    [HttpGet("model-stats")]
    public async Task<IActionResult> GetModelStats()
    {
        try
        {
            var content = await _mlService.GetMetricsAsync();
            return Content(content, "application/json");
        }
        catch (Exception ex)
        {
            return StatusCode(503, new { available = false, message = "ML Service metrics unavailable: " + ex.Message });
        }
    }
}
