using Microsoft.EntityFrameworkCore;

namespace Backend.Data;


public class ScanLog
{
    public int Id { get; set; }
    public string Query { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public bool IsSqlInjection { get; set; }
    public DateTime ScannedAt { get; set; } = DateTime.UtcNow;
}
public class AppDbContext : DbContext
{   
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) {}
    public DbSet<ScanLog> ScanLogs => Set<ScanLog>();
}
