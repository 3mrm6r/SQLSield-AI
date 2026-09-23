import { HttpClient } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { Observable } from "rxjs";

export interface AnalyzeResponse {
  id?: number;
  query: string;
  label: string;
  confidence: number;
  isSqlInjection: boolean;
  scannedAt: string;
}

export interface ModelBenchmarkItem {
  Model: string;
  Accuracy: number;
  Precision: number;
  Recall: number;
  F1: number;
  'ROC-AUC': number;
  FPR: number;
  FNR: number;
  'Train Time (s)': number;
  'CV F1 Mean': number;
  'CV F1 Std': number;
}

export interface EvaluationMetricsData {
  raw_text?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  fpr?: number;
  fnr?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  tp?: number;
  test_samples?: number;
  total_time?: string;
  avg_query_time_ms?: string;
}

export interface FeatureImportanceItem {
  token: string;
  importance: number;
}

export interface ModelReportsResponse {
  available: boolean;
  message?: string;
  best_model?: string;
  models?: ModelBenchmarkItem[];
  evaluation?: EvaluationMetricsData;
  top_features?: FeatureImportanceItem[];
}

@Injectable({
  providedIn: 'root'
})
export class DetectionService {
  private readonly baseUrl = 'http://localhost:5000/api/detection';

  constructor(private http: HttpClient) {}

  analyze(query: string): Observable<AnalyzeResponse> {
    return this.http.post<AnalyzeResponse>(`${this.baseUrl}/analyze`, { query });
  }

  getHistory(): Observable<AnalyzeResponse[]> {
    return this.http.get<AnalyzeResponse[]>(`${this.baseUrl}/history`);
  }

  getModelStats(): Observable<ModelReportsResponse> {
    return this.http.get<ModelReportsResponse>(`${this.baseUrl}/model-stats`);
  }
}