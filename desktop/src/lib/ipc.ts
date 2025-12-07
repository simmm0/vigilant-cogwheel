import { invoke } from "@tauri-apps/api/tauri";
import { AnnealingPreset } from "../types";

export async function runTraining(payload: AnnealingPreset & { authToken?: string }) {
  return invoke<string>("run_training", { payload });
}

export async function stopTraining(jobId: string) {
  return invoke<void>("stop_training", { jobId });
}

export async function fetchLogs(jobId: string) {
  return invoke<string>("tail_logs", { jobId });
}

