import { Component, OnInit, signal, computed } from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  DetectionService,
  AnalyzeResponse,
  ModelReportsResponse
} from './services/detection.service';

interface PresetQuery {
  label: string;
  category: 'malicious' | 'benign';
  sql: string;
}

export type MetricsTab = 'comparison' | 'matrix' | 'features' | 'raw';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, DatePipe, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit {
  queryInput = signal<string>('');
  isLoading = signal<boolean>(false);
  errorMessage = signal<string>('');
  currentResult = signal<AnalyzeResponse | null>(null);
  history = signal<AnalyzeResponse[]>([]);
  modelReports = signal<ModelReportsResponse | null>(null);
  activeTab = signal<MetricsTab>('comparison');

  charCount = computed(() => this.queryInput().length);
  hasQuery = computed(() => this.queryInput().trim().length > 0);

  readonly presets: PresetQuery[] = [
    {
      label: 'Benign Customer Lookup',
      category: 'benign',
      sql: 'SELECT email, name FROM customers WHERE id = 100;'
    },
    {
      label: 'Auth Bypass (Tautology)',
      category: 'malicious',
      sql: "admin' OR '1'='1"
    },
    {
      label: 'Stacked Query (Drop DB)',
      category: 'malicious',
      sql: '1; DROP TABLE users; --'
    },
    {
      label: 'UNION Data Exfiltration',
      category: 'malicious',
      sql: "1' UNION SELECT NULL, username, password FROM users --"
    },
    {
      label: 'Product Search Filter',
      category: 'benign',
      sql: "SELECT * FROM products WHERE category = 'electronics' AND price < 500;"
    },
    {
      label: 'Time-Based Blind Injection',
      category: 'malicious',
      sql: "' OR SLEEP(5) --"
    }
  ];

  constructor(private detectionService: DetectionService) {}

  ngOnInit(): void {
    this.fetchHistory();
    this.loadStats();
  }

  setPreset(sql: string): void {
    this.queryInput.set(sql);
    this.errorMessage.set('');
  }

  clearInput(): void {
    this.queryInput.set('');
    this.currentResult.set(null);
    this.errorMessage.set('');
  }

  setActiveTab(tab: MetricsTab): void {
    this.activeTab.set(tab);
  }

  fetchHistory(): void {
    this.detectionService.getHistory().subscribe({
      next: (data) => {
        this.history.set(data);
      },
      error: (err) => {
        console.warn('Backend history service unavailable:', err);
      }
    });
  }

  analyzeQuery(): void {
    const text = this.queryInput().trim();
    if (!text) return;

    this.isLoading.set(true);
    this.errorMessage.set('');
    this.currentResult.set(null);

    this.detectionService.analyze(text).subscribe({
      next: (res) => {
        this.currentResult.set(res);
        this.isLoading.set(false);
        this.fetchHistory();
      },
      error: (err) => {
        this.isLoading.set(false);
        this.errorMessage.set(
          'Failed to connect to backend server. Please verify ASP.NET Core API (port 5000) and ML microservice (port 8000) are running.'
        );
      }
    });
  }

  loadStats(): void {
    this.detectionService.getModelStats().subscribe({
      next: (data) => {
        this.modelReports.set(data);
      },
      error: (err) => {
        console.warn('Could not load model reports:', err);
      }
    });
  }
}
