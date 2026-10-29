// src/lib/api.ts — Typed API client for QPredRNN FastAPI backend
const BASE = "/api";

export interface CanonicalModel {
  id: string;
  name: string;
  display_name: string;
  family: string;
  description: string;
  parameter_count: number;
  quantum_enabled: boolean;
  n_qubits?: number;
  circuit_depth?: number;
  simulator_backend?: string;
  input_frames: number;
  output_frames: number;
  implementation_file: string;
  default_run_id: string;
}

export interface Run {
  id: string;
  name: string;
  display_name?: string;
  model_type: string;
  category?: "primary" | "primary_final" | "pilot" | "smoke_test" | "profiling" | "experimental";
  status?: string;
  description?: string;
  has_best_ckpt: boolean;
  has_last_ckpt: boolean;
  best_val_mse: number | null;
  last_epoch: number | null;
  total_epochs: number | null;
  has_history: boolean;
  history_entries: number;
  path: string;
  config?: Record<string, unknown>;
}

export interface LiveTrainingStatus {
  run_id: string;
  display_name: string;
  status: "NOT_STARTED" | "INITIALIZING" | "RUNNING" | "COMPLETED";
  target_epochs: number;
  completed_epochs: number;
  current_epoch: number | null;
  train_loss: number | null;
  val_loss: number | null;
  best_val_loss: number | null;
  best_val_epoch: number | null;
  last_update_ts: number | null;
  has_best_ckpt: boolean;
  history_length: number;
}

export interface LiveRunHistory {
  run_id: string;
  display_name: string;
  history: HistoryEntry[];
  status: string;
  target_epochs: number;
  completed_epochs: number;
}

export interface CheckpointItem {
  file_path: string;
  run_id: string;
  filename: string;
  architecture: string;
  model_type: string;
  run_type: string;
  classification: string;
  file_size_bytes: number;
  sha256: string;
  is_best: boolean;
  is_last: boolean;
  recorded_epoch: number | null;
  recorded_global_step: number | null;
  configured_epochs: number | null;
  history_epochs: number;
  best_val_loss: number | null;
  last_val_loss: number | null;
  train_status: string;
  parameter_count: number | null;
  load_status: string;
  load_ok: boolean;
  forward_status: string;
  forward_ok: boolean;
  output_shape: number[] | null;
  is_finite: boolean;
  quantum_path_verified: boolean;
  duplicate_of: string | null;
}

export interface BenchmarkMetrics {
  mse: number;
  mae: number;
  psnr: number;
  ssim: number;
  latency_ms_per_window: number;
  total_eval_time_s: number;
}

export interface BenchmarkResult {
  model_id: string;
  display_name: string;
  architecture_id: string;
  architecture: string;
  run_id: string;
  checkpoint_path: string | null;
  checkpoint_sha256: string | null;
  checkpoint_epoch: number | null;
  evaluation_split: string;
  evaluated_windows: number;
  total_available_windows: number;
  device: string;
  evaluation_timestamp: string;
  metrics: BenchmarkMetrics;
  per_horizon: {
    mse: number[];
    mae: number[];
    psnr: number[];
    ssim: number[];
  };
  distribution?: Record<string, unknown>;
  status: string;
  notes?: string;
}

export interface SystemInfo {
  python: string;
  pytorch: string;
  cuda_available: boolean;
  gpu_name: string | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
  cached_models: number;
  kth_available: boolean;
  toy_available: boolean;
}

export interface DatasetInfo {
  id: string;
  name: string;
  available: boolean;
  path?: string;
  note?: string;
}

export interface SplitInfo {
  dataset: string;
  split: string;
  num_clips: number;
  input_frames: number;
  output_frames: number;
  stride: number;
  num_sequences: number;
  sequence_ids: string[];
}

export interface PerFrameMetrics {
  mse: number[];
  mae: number[];
  psnr: number[];
  ssim: number[];
}

export interface Metrics {
  mse: number;
  mae: number;
  psnr: number;
  ssim: number;
  per_frame: PerFrameMetrics;
}

export interface TensorInfo {
  input_shape: number[];
  gt_shape: number[];
  pred_shape: number[];
  pred_std: number;
  pred_range: [number, number];
}

export interface PredictResult {
  run_id: string;
  checkpoint: string;
  checkpoint_path: string;
  checkpoint_epoch: number | null;
  checkpoint_best_val_loss: number | null;
  dataset: string;
  split: string;
  clip_index: number;
  seq_desc: string;
  device_used: string;
  latency_ms: number;
  input_frames: string[]; // base64 PNG
  gt_frames: string[];
  pred_frames: string[];
  error_frames: string[];
  metrics: Metrics;
  tensor_info: TensorInfo;
}

export interface HistoryEntry {
  epoch: number;
  train_loss: number;
  val_loss: number;
  best_val_loss?: number;
  decouple_loss?: number;
  r_eta?: number;
  eta?: number;
  is_best?: boolean;
  global_step?: number;
}

export interface PredictRequest {
  run_id: string;
  checkpoint: "best" | "last";
  dataset: "kth" | "toy";
  split: "train" | "val" | "test";
  clip_index: number;
  device: "cuda" | "cpu";
}

export interface CompareModelItem {
  run_id: string;
  checkpoint?: "best" | "last";
  display_name?: string;
}

export interface CompareRequest {
  models: CompareModelItem[];
  dataset: "kth" | "toy";
  split: "train" | "val" | "test";
  clip_index: number;
  device: "cuda" | "cpu";
}

export interface ComparedModelResult {
  run_id: string;
  display_name: string;
  checkpoint: string;
  success: boolean;
  error?: string;
  checkpoint_path?: string;
  checkpoint_epoch?: number | null;
  checkpoint_best_val_loss?: number | null;
  latency_ms?: number;
  pred_frames?: string[];
  error_frames?: string[];
  metrics?: Metrics;
}

export interface CompareResponse {
  dataset: string;
  split: string;
  clip_index: number;
  seq_desc: string;
  input_frames: string[];
  gt_frames: string[];
  models: ComparedModelResult[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; timestamp: number }>("/health"),
  system: () => request<SystemInfo>("/system"),
  models: () => request<CanonicalModel[]>("/models"),
  experiments: () => request<Run[]>("/experiments"),
  runs: () => request<Run[]>("/runs"),
  run: (id: string) => request<Run>(`/runs/${id}`),
  history: (id: string) => request<HistoryEntry[]>(`/runs/${id}/history`),
  checkpoints: () => request<CheckpointItem[]>("/checkpoints"),
  benchmarks: () => request<BenchmarkResult[]>("/benchmarks"),
  datasets: () => request<DatasetInfo[]>("/datasets"),
  splitInfo: (ds: string, split: string, stride = 2) =>
    request<SplitInfo>(`/datasets/${ds}/splits/${split}/info?stride=${stride}`),
  predict: (body: PredictRequest) =>
    request<PredictResult>("/predict", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: CompareRequest) =>
    request<CompareResponse>("/compare", { method: "POST", body: JSON.stringify(body) }),
  metricsInfo: () => request<Record<string, unknown>>("/metrics"),
  trainingLive: () => request<LiveTrainingStatus[]>("/training/live"),
  trainingLiveRun: (runId: string) => request<LiveRunHistory>(`/training/live/${runId}`),
};
